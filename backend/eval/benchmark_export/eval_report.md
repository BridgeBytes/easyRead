# EasyRead Tier 1 Benchmark Report

Generated: 2026-02-26 11:14 UTC

## GPU Environment

| Field | Value |
|---|---|
| GPU Name | Tesla T4 |
| CUDA Version | 13.0 |
| PyTorch Version | 2.10.0a0+b558c986e8.nv25.11 |
| Device Count | 1 |

## Summary Statistics

| Metric | Value |
|---|---|
| Images Scored | 10 |
| Mean EasyRead Score | 0.4755 |
| Std Dev | 0.0420 |
| Min | 0.4127 |
| Max | 0.5248 |

## Comparison to Paper Baselines

| Baseline | Paper Score | Our Mean | Delta |
|---|---|---|---|
| SD v1.5 (Table 1) | 0.4005 | 0.4755 | +0.0750 |
| LoRA Finetuned (Table 1) | 0.4697 | 0.4755 | +0.0058 |

## Per-Image Results

| Image | Palette | Edge | Saliency | Contrast | Stroke | Centering | **EasyRead** | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Apple_company_logo.png | 0.0695 | 0.7814 | 0.5952 | 0.3088 | 0.7152 | 0.5841 | **0.4750** | 0.38 |
| a_socker_player_kicking_a_ball,_background_color_green,_hair_color_dark_brown.png | 0.0048 | 0.5229 | 0.6700 | 0.1290 | 0.7152 | 0.7980 | **0.4127** | 0.27 |
| chat_message_bubbles.png | 0.0011 | 0.4846 | 0.7561 | 0.8811 | 0.7152 | 0.7476 | **0.5248** | 0.29 |
| Accessibility_symbol.png | 0.0183 | 0.6522 | 0.5582 | 0.3751 | 0.7152 | 0.8320 | **0.4655** | 0.3 |
| a_Food_pyramid.png | 0.0001 | 0.5251 | 0.7130 | 0.7548 | 0.7152 | 0.2456 | **0.4571** | 0.28 |
| chat_message_bubbles.png | 0.2636 | 0.7251 | 0.6444 | 0.1330 | 0.7152 | 0.7814 | **0.5129** | 0.23 |
| Apple_company_logo.png | 0.0421 | 0.5960 | 0.4480 | 0.8469 | 0.7152 | 0.8289 | **0.5141** | 0.22 |
| Apple_computer_logo.png | 0.2636 | 0.7593 | 0.7387 | 0.2389 | 0.7152 | 0.5021 | **0.5219** | 0.22 |
| a_socker_player_kicking_a_ball,_background_color_green,_hair_color_dark_brown.png | 0.0029 | 0.5397 | 0.8841 | 0.2320 | 0.7152 | 0.3183 | **0.4152** | 0.29 |
| car_engine_inspection.png | 0.0007 | 0.3599 | 0.4184 | 0.8818 | 0.7152 | 0.8168 | **0.4561** | 0.31 |

## Chart

![EasyRead Benchmark Chart](eval_scores_chart.png)
