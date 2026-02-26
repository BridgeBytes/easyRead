#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 1 Benchmark: EasyRead scoring against paper baselines.

Validates GPU, collects up to 10 PNGs from backend/temp/, scores each using
the EasyRead metrics, and writes eval_scores.csv, eval_scores_chart.png, and
eval_report.md to /backend/eval/.

Usage (inside container):
  python /backend/eval/tier1_benchmark.py
"""

import sys
import os
import time
import csv
import subprocess
from pathlib import Path
from datetime import datetime

# ---------- Paths ----------
EVAL_DIR    = Path(__file__).resolve().parent   # /backend/eval/
BACKEND_DIR = EVAL_DIR.parent                   # /backend/
TEMP_DIR    = BACKEND_DIR / "temp"              # /backend/temp/
OUTPUT_DIR  = EVAL_DIR                          # /backend/eval/

MAX_IMAGES     = 10
PAPER_BASELINE = 0.4005   # SD v1.5 from paper Table 1
PAPER_LORA     = 0.4697   # LoRA finetuned from paper Table 1


# ---------- 1. Dependency bootstrap ----------

def _ensure_dep(package: str, import_name: str = None) -> None:
    """Install package via pip if it cannot be imported (silent on success)."""
    mod = import_name or package
    try:
        __import__(mod)
    except Exception:
        print(f"[bootstrap] Installing missing dependency: {package}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package]
        )


_ensure_dep("scipy")
_ensure_dep("scikit-image", "skimage")
_ensure_dep("matplotlib")
# Use headless variant — avoids libGL.so.1 dependency absent in server containers
_ensure_dep("opencv-python-headless", "cv2")


# ---------- 2. GPU validation ----------

def validate_gpu() -> dict:
    """Assert a CUDA GPU is visible; exit with error if not. Returns env info."""
    try:
        import torch
    except ImportError:
        print("[ERROR] torch is not importable — cannot validate GPU.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found. Check GPU passthrough:")
        print("        docker-compose exec backend nvidia-smi")
        sys.exit(1)

    gpu_name      = torch.cuda.get_device_name(0)
    cuda_version  = torch.version.cuda
    torch_version = torch.__version__
    device_count  = torch.cuda.device_count()

    print(
        f"[GPU] OK — {gpu_name}  CUDA {cuda_version}"
        f"  PyTorch {torch_version}  devices={device_count}"
    )
    return {
        "gpu_name":      gpu_name,
        "cuda_version":  cuda_version,
        "torch_version": torch_version,
        "device_count":  device_count,
    }


# ---------- 3. Image collection ----------

def collect_images(temp_dir: Path, max_images: int = MAX_IMAGES) -> list:
    """
    Breadth-first collection: take 1 PNG per UUID sub-directory first, then
    backfill from dirs that have more. Returns up to max_images Path objects.
    """
    if not temp_dir.exists():
        print(f"[ERROR] TEMP_DIR not found: {temp_dir}")
        sys.exit(1)

    subdirs = sorted(d for d in temp_dir.iterdir() if d.is_dir())
    buckets = [sorted(d.glob("*.png")) for d in subdirs]
    buckets = [b for b in buckets if b]  # drop empty dirs

    if not buckets:
        print(f"[ERROR] No PNG images found under {temp_dir}")
        sys.exit(1)

    selected: list = []

    # Round 1: one image per UUID dir
    for bucket in buckets:
        if len(selected) >= max_images:
            break
        selected.append(bucket[0])

    # Backfill: extras from dirs that have more than one PNG
    if len(selected) < max_images:
        for bucket in buckets:
            for png in bucket[1:]:
                if len(selected) >= max_images:
                    break
                selected.append(png)
            if len(selected) >= max_images:
                break

    print(
        f"[collect] {len(selected)} images selected"
        f" (from {len(buckets)} UUID dirs)"
    )
    return selected[:max_images]


# ---------- 4. Scoring ----------

# Add EVAL_DIR to sys.path so easyread_metrics is importable when run directly
sys.path.insert(0, str(EVAL_DIR))
from easyread_metrics import compute_metrics, compute_easyread_components_from_raw  # noqa: E402

SUB_SCORES = [
    "palette_score",
    "edge_score",
    "saliency_score",
    "contrast_score",
    "stroke_score",
    "centering_score",
]


def score_images(image_paths: list) -> list:
    """Score each image; skip (with warning) any that fail. Returns list of dicts."""
    results = []
    n = len(image_paths)
    for i, path in enumerate(image_paths, start=1):
        print(f"[{i}/{n}] {path.name} ... ", end="", flush=True)
        t0 = time.time()
        try:
            raw  = compute_metrics(str(path))
            comp = compute_easyread_components_from_raw(raw)
            elapsed = time.time() - t0
            row = {
                "image":         path.name,
                "path":          str(path),
                "elapsed_s":     round(elapsed, 2),
                **{k: round(comp[k], 4) for k in SUB_SCORES},
                "easyread_score": round(comp["easyread_score"], 4),
            }
            results.append(row)
            print(f"EasyRead={comp['easyread_score']:.4f}  ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"WARN — skipped ({exc})")
    return results


# ---------- 5a. CSV ----------

def write_csv(results: list, out_path: Path) -> None:
    fieldnames = ["image", "path", "elapsed_s"] + SUB_SCORES + ["easyread_score"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"[output] CSV     -> {out_path}")


# ---------- 5b. Chart ----------

def write_chart(results: list, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n           = len(results)
    x           = np.arange(n)
    short_names = [r["image"][:22] for r in results]

    bar_w   = 0.12
    n_subs  = len(SUB_SCORES)
    offsets = np.linspace(-(n_subs - 1) / 2, (n_subs - 1) / 2, n_subs)
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("EasyRead Tier 1 Benchmark", fontsize=14, fontweight="bold")

    # --- Left: grouped sub-scores ---
    for sub, color, offset in zip(SUB_SCORES, colors, offsets):
        vals = [r[sub] for r in results]
        ax1.bar(x + offset * bar_w, vals, bar_w,
                label=sub.replace("_score", ""), color=color, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=40, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Sub-score (0–1)")
    ax1.set_title("Sub-scores per Image")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # --- Right: composite + baselines ---
    composite = [r["easyread_score"] for r in results]
    ax2.bar(x, composite, color="#4C72B0", alpha=0.85, label="Our images")
    ax2.axhline(PAPER_BASELINE, color="#C44E52", linestyle="--", linewidth=1.8,
                label=f"SD v1.5 baseline ({PAPER_BASELINE})")
    ax2.axhline(PAPER_LORA, color="#55A868", linestyle="--", linewidth=1.8,
                label=f"LoRA finetuned ({PAPER_LORA})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=40, ha="right", fontsize=8)
    ax2.set_ylim(0, max(0.65, max(composite) * 1.15))
    ax2.set_ylabel("EasyRead Score")
    ax2.set_title("Composite EasyRead Score vs Paper Baselines")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[output] Chart   -> {out_path}")


# ---------- 5c. Markdown report ----------

def write_report(
    results: list, gpu_info: dict, out_path: Path, chart_path: Path
) -> None:
    import numpy as np

    if not results:
        out_path.write_text("# Tier 1 Benchmark\n\nNo images scored.\n")
        return

    scores     = [r["easyread_score"] for r in results]
    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    d_baseline = mean_score - PAPER_BASELINE
    d_lora     = mean_score - PAPER_LORA
    timestamp  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# EasyRead Tier 1 Benchmark Report",
        "",
        f"Generated: {timestamp}",
        "",
        "## GPU Environment",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| GPU Name | {gpu_info['gpu_name']} |",
        f"| CUDA Version | {gpu_info['cuda_version']} |",
        f"| PyTorch Version | {gpu_info['torch_version']} |",
        f"| Device Count | {gpu_info['device_count']} |",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Images Scored | {len(results)} |",
        f"| Mean EasyRead Score | {mean_score:.4f} |",
        f"| Std Dev | {std_score:.4f} |",
        f"| Min | {min(scores):.4f} |",
        f"| Max | {max(scores):.4f} |",
        "",
        "## Comparison to Paper Baselines",
        "",
        "| Baseline | Paper Score | Our Mean | Delta |",
        "|---|---|---|---|",
        f"| SD v1.5 (Table 1) | {PAPER_BASELINE} | {mean_score:.4f} | {d_baseline:+.4f} |",
        f"| LoRA Finetuned (Table 1) | {PAPER_LORA} | {mean_score:.4f} | {d_lora:+.4f} |",
        "",
        "## Per-Image Results",
        "",
        "| Image | Palette | Edge | Saliency | Contrast"
        " | Stroke | Centering | **EasyRead** | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r['image']}"
            f" | {r['palette_score']:.4f}"
            f" | {r['edge_score']:.4f}"
            f" | {r['saliency_score']:.4f}"
            f" | {r['contrast_score']:.4f}"
            f" | {r['stroke_score']:.4f}"
            f" | {r['centering_score']:.4f}"
            f" | **{r['easyread_score']:.4f}**"
            f" | {r['elapsed_s']} |"
        )

    lines += [
        "",
        "## Chart",
        "",
        f"![EasyRead Benchmark Chart]({chart_path.name})",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[output] Report  -> {out_path}")


# ---------- Main ----------

def main() -> None:
    print("=" * 60)
    print("  EasyRead Tier 1 Benchmark")
    print("=" * 60)

    gpu_info    = validate_gpu()
    image_paths = collect_images(TEMP_DIR)
    results     = score_images(image_paths)

    if not results:
        print("[ERROR] All images failed to score.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path    = OUTPUT_DIR / "eval_scores.csv"
    chart_path  = OUTPUT_DIR / "eval_scores_chart.png"
    report_path = OUTPUT_DIR / "eval_report.md"

    write_csv(results, csv_path)
    write_chart(results, chart_path)
    write_report(results, gpu_info, report_path, chart_path)

    import numpy as np
    scores     = [r["easyread_score"] for r in results]
    mean_score = float(np.mean(scores))

    print()
    print("=" * 60)
    print(f"  Images scored       : {len(results)}")
    print(f"  Mean EasyRead Score : {mean_score:.4f}")
    print(f"  Paper SD v1.5       : {PAPER_BASELINE}"
          f"  (delta {mean_score - PAPER_BASELINE:+.4f})")
    print(f"  Paper LoRA          : {PAPER_LORA}"
          f"  (delta {mean_score - PAPER_LORA:+.4f})")
    print("=" * 60)
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
