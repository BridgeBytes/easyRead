# Finetuning Diffusion Models for EasyRead-Style Image Generation

**Course:** 263-3300-10L Data Science Lab  
**Date:** December 17, 2025, Zurich, Switzerland  
**Team 4**

**Authors:**  
- Yanis Merzouki* — Department of Computer Science, ETH Zurich  
- Nicolas Dickenmann* — Department of Computer Science, ETH Zurich  
- Thy Nowak-Tran — UNICEF Digital Centre of Excellence  
- Gerda Binder — UNICEF Digital Centre of Excellence  
- Sonia Laguna Cillero — ETH AI Center, ETH Zurich  
- Emanuele Palumbo — ETH AI Center, ETH Zurich  
- Julia Vogt — ETH AI Center, ETH Zurich  

*Equal contribution*

**Code:** [https://github.com/easyread-dsl/easyread_project](https://github.com/easyread-dsl/easyread_project) (MIT License)

---

## Abstract

EasyRead pictograms are designed to support cognitive accessibility through minimal visual complexity, strong contrast, and clear semantic focus. However, standard diffusion models tend to produce cluttered or stylistically inconsistent outputs that do not satisfy these constraints. This work presents a unified pipeline for generating accessibility-oriented pictograms by finetuning Stable Diffusion 1.5 with LoRA adapters on a curated corpus combining ARASAAC, LDS, OpenMoji, and custom augmentations. We generate natural-language prompts using BLIP captioning, compute objective EasyRead metrics, and track both semantic alignment and accessibility quality during training. Our results show that diffusion models can be steered toward consistent EasyRead-style outputs, highlighting the potential of generative models as practical tools for producing cognitively accessible visual communication.

**Keywords:** Diffusion Models, Generative AI, LoRA Finetuning, Cognitive Accessibility, Pictogram Generation, EasyRead Metrics, Visual Communication

---

## 1. Introduction

Visual symbols and pictograms are essential tools for communicating information in a concise and accessible manner. They support users with diverse linguistic or cognitive backgrounds, and are widely deployed in education, healthcare, and public services.

Among existing design frameworks, EasyRead emphasizes clarity through minimal detail, strong contrast, and unambiguous depiction of key semantic elements. Creating such visuals at scale, however, remains labor-intensive and requires domain expertise.

Recent progress in generative diffusion models has enabled high-quality image synthesis from natural-language prompts. These models are versatile and widely deployable, yet they typically produce visually rich outputs that conflict with the stylistic constraints required for cognitive accessibility. This gap raises an important question: **can powerful generative models be adapted to consistently produce simplified, accessibility-aligned pictograms?**

This work explores this question via the following contributions:
- Curating a unified corpus of pictogram datasets
- Generating natural-language prompts using BLIP-based captioning
- Computing objective metrics that quantify EasyRead properties (palette simplicity, contour clarity, contrast)
- Finetuning Stable Diffusion 1.5 using LoRA adapters, enabling stylistic adaptation without modifying full model weights

---

## 2. Related Work

Diffusion models can be efficiently adapted to new visual domains through lightweight finetuning techniques such as LoRA, DreamBooth, and Textual Inversion. These methods provide strong style specialization while keeping the base model frozen, though none have been applied to accessibility-oriented visual design.

Prior work on pictogram and icon generation includes Piconet, symbolic style simplification, ICONATE, IconGAN, and Auto-Icon. These systems focus on improving aesthetics, semantic consistency, or developer workflows, but do not incorporate cognitive accessibility constraints.

Human–computer interaction research shows that stylistic variations affect recognition accuracy and cognitive load, and that aesthetics, complexity, and concreteness jointly shape icon usability. Work in cognitive accessibility further emphasizes the need for measurable criteria that promote clarity and ease of understanding.

**To our knowledge, no prior work combines text-to-image diffusion with explicit, pixel-based EasyRead metrics to guide or evaluate pictogram generation.**

---

## 3. Dataset

The training corpus combines three publicly available pictogram and icon datasets: **OpenMoji**, **ARASAAC**, and the **Learning Disability Service (LDS)** set. These collections differ significantly in style, semantic granularity, and metadata quality, requiring a unified preprocessing pipeline.

### 3.1 Datasets Used

- **OpenMoji:** An open-source emoji set containing thousands of vector-based icons with consistent styling but limited semantic context.
- **ARASAAC:** A large, curated set of pictograms used in augmentative and alternative communication.
- **LDS (Learning Disability Service):** A set of pictograms used in accessibility contexts.

> Note: ICON645 was evaluated but excluded from the final configuration due to heterogeneous visual style and weak semantic correspondence to EasyRead pictograms.

### 3.2 Caption Limitations and Prompt Generation

Across all three datasets, the textual metadata was not detailed enough to serve as high-quality prompts for diffusion training. Titles and keyword lists do not reflect visible attributes such as object count, poses, colors, or relationships between elements.

To overcome this, natural-language descriptions are generated for every image using the **BLIP (Bootstrapping Language-Image Pre-training) Image Captioning model (Large)**. Prompts are produced directly from raw pixels, ignoring original dataset labels. Post-processing steps include:
- Removing boilerplate prefixes (e.g., "an image of")
- Eliminating the word "cartoon" (redundant for icon datasets)
- Normalizing whitespace and capitalizing the final text

**Example caption refinements:**

| Original Tag | BLIP-Generated Caption |
|---|---|
| "friends" | "Three people standing together with their arms around each other" |
| "water" | "A bottle of water next to a glass of water" |

### 3.3 ARASAAC Augmentation

The ARASAAC dataset is augmented by exploiting customization features on the official ARASAAC website. Each pictogram can be rendered with different background colors, skin tones, and hair colors. These parameters are systematically varied to increase diversity while preserving the EasyRead aesthetic, enabling controllability over these properties at inference time.

Each training prompt is appended with:
```
; background color: <color>; skin color: <color>; hair color: <color>
```

**Customization options:**
- **Background colors:** red, green, blue, yellow, black, white (also accepts HEX codes)
- **Skin colors:** white, black, asian, mulatto, aztec
- **Hair colors:** blonde, brown, darkBrown, gray, darkGray, red, black

Images without people receive these labels too, allowing the model to learn to ignore irrelevant properties.

---

## 4. EasyRead Metrics

The EasyRead metric suite quantifies how well an image follows "easy-to-understand pictogram" conventions. These metrics target visual properties that make images accessible: a small and clean color palette, simple shapes, a clear focal object, strong foreground–background separation, and consistent strokes.

### 4.1 Design Principles

- **Low visual clutter:** the image should not contain many different colors or fine-grained noisy details.
- **Simple geometry:** the main shapes should be easy to parse, with clear, relatively smooth outlines.
- **Clear focus:** there should be a visually dominant object rather than many competing elements.
- **Strong separation:** the foreground should stand out clearly from the background in terms of perceived brightness.
- **Stable layout:** the main content should be roughly centered and occupy a substantial, but not overwhelming, part of the frame.
- **Consistent strokes:** outlines should have a reasonably uniform, readable thickness relative to the image size.

### 4.2 Composite EasyRead Score

```
EasyReadScore = 0.25·s_palette + 0.20·s_edges + 0.15·s_saliency
              + 0.15·s_contrast + 0.15·s_stroke + 0.10·s_centering
```

Each `s ∈ [0, 1]` is a normalized metric score. EasyRead pictures steer towards 1; complex images towards 0.

### 4.3 Palette Complexity (`s_palette`, weight: 0.25)

Measures how many distinct colors the image effectively uses. Colors are coarsened by snapping each RGB channel to discrete steps (multiples of 8) to merge similar shades. Only colors covering at least ~0.1% of the image area are counted. The raw count is transformed by a smooth **decreasing** function — small palettes score high, larger ones are penalized. This is one of the best-performing metrics for separating simple pictograms from realistic images.

### 4.4 Edge Density (`s_edges`, weight: 0.20)

Captures how busy the line work is. The image is resized to a fixed reference width, converted to grayscale, and edges are detected using **Canny edge detection**. The metric is the fraction of pixels classified as edges. Low edge densities map close to 1; higher values are progressively penalized. Works well for the evaluation pipeline.

### 4.5 Saliency Concentration (`s_saliency`, weight: 0.15)

Quantifies how much visually important content is concentrated in a single main object. Computed using **spectral residual saliency**. The smallest set of pixels whose saliency mass sums to 20% of the total is identified, and the fraction of that mass contained in the single largest connected component is measured. Close to 1 if attention is focused on one blob; smaller if split across many regions.

> Note: Realistic images scored better than pictograms on this metric — theorized to be due to transparent backgrounds in the dataset.

### 4.6 Foreground–Background Contrast (`s_contrast`, weight: 0.15)

Measures how clearly the main content stands out from the background. The image is converted to **CIE LAB color space** and the lightness channel is isolated. The foreground mask reuses the salient region from the saliency step; background is its complement. The raw metric is the absolute difference between robust mean lightness values of foreground and background.

### 4.7 Stroke Thickness (`s_stroke`, weight: 0.15)

Estimates a typical stroke thickness for pictogram outlines. The image is binarized, and a **distance transform** is applied: each foreground pixel stores its distance to the nearest background pixel. The median of sampled half-thickness values is doubled to estimate stroke width, then divided by image height to yield a relative stroke thickness. Scores are highest when stroke thickness falls within a target comfortable range.

### 4.8 Centering Error (`s_centering`, weight: 0.10)

Captures the layout of the main content. The centroid of the foreground (salient) mask is computed in normalized coordinates. The centering error is:
```
centering_error = max(|cx - 0.5|, |cy - 0.5|)
```
Zero for a perfectly centered subject. The normalized score is a **decreasing** function — centered layouts are rewarded, off-center ones penalized.

### 4.9 Normalization Functions

- **Decreasing** `exp(-k·x)`: palette size (k=2.0), edge density (k=2.5), centering error (k=3.0)
- **Increasing saturating** `1 - exp(-k·x)`: saliency concentration (k=4.0), contrast (k=3.0)
- **Centered Gaussian** `exp(-(x-μ)²/(2σ²))`: stroke thickness (μ=0.015, σ=0.006)

---

## 5. Model and Training

### 5.1 Method

Base model: **Stable Diffusion 1.5** (`runwayml/stable-diffusion-v1-5`)

Architecture:
- UNet denoiser conditioned on CLIP text embeddings
- Variational Autoencoder (VAE) for latent space mapping
- CLIP tokenizer and text encoder remain **frozen** throughout training

**LoRA (Low-Rank Adaptation)** is applied to the attention layers of the UNet (key, query, value, and output projections), enabling style learning without modifying full model weights.

### 5.2 Experimental Details

| Parameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA scaling parameter | 16 |
| Image resolution | 512 × 512 |
| GPU | NVIDIA T4 |
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Precision | Mixed |

Every caption is prefixed with a dedicated **instance token** to activate the learned style at inference. For evaluation, the instance token is prepended for the finetuned model; "A pictogram of" is prepended for the baseline.

### 5.3 Evaluation Setup

- **55 diverse prompts** covering objects, human activities, abstract concepts, and multi-object scenes
- All prompts structured with ARASAAC augmentation format (background, skin, hair attributes)
- **5 images generated per prompt** for both base and finetuned models (255 images per model)
- Paired comparison: same random seeds across models

---

## 6. Results

### 6.1 Quantitative Results

| Model | EasyRead Score ↑ | CLIP Score ↑ |
|---|---|---|
| Baseline (SD v1.5) | 0.4005 | 24.3257 |
| **Ours (LoRA finetuned)** | **0.4697** | **31.1542** |

The finetuned model achieves a +17% improvement in EasyRead score and a +28% improvement in CLIP similarity. This indicates finetuning not only enhances visual clarity but also strengthens semantic alignment with the user's prompt.

---

## 7. Discussion

### Strengths
- Strong adherence to background, skin tone, and hair color instructions
- High-quality human figures and faces in simple scenes
- Effectively ignores irrelevant instructions (e.g., skin/hair color when no person is present)
- Often rivals or surpasses SOTA models (Global Symbols, Nano Banana Pro) on instruction following for pictogram-specific tasks

### Known Limitations
- **Seed variance:** Outputs vary meaningfully across seeds — multiple samples (typically 4) should be generated and the best selected
- **Background interpretation:** Sometimes generates contextual/scenic backgrounds instead of flat color fills
- **Complex scenes:** Struggles with detail abstraction in multi-person scenes; faces and key features can degrade
- **Skin tone ≠ ethnicity:** ARASAAC skin tone augmentation is a proxy and does not fully represent ethnicity

### Comparison to External Models

| | SD v1.5 | Ours (LoRA) | Global Symbols | Nano Banana Pro |
|---|---|---|---|---|
| EasyRead style | ✗ | ✓ | ✓ | Inconsistent |
| Instruction following | Poor | Strong | Good | Variable |
| Quantitative eval | Baseline | Best | Not measured | Not measured |

### Future Directions
- Incorporate **differentiable EasyRead metrics** directly into the loss function during training
- Add a **style-consistency objective** (e.g., neural style loss with dataset-level Gram statistics)
- Finetune EasyRead score hyperparameters or replace metrics with weaker discriminative power

---

## 8. Conclusion

Targeted finetuning of Stable Diffusion on curated pictogram datasets enables reliable generation of EasyRead-style images that prioritize accessibility through visual simplicity and semantic clarity. The approach introduces controllable generation of background colors, skin tones, and hair colors while maintaining strong generalization to unseen concepts. By combining domain-specific training data with BLIP-enhanced captions and automated EasyRead metrics, this framework significantly outperforms standard diffusion models in producing visually clear, accessibility-focused imagery.

---

## Appendix A: Dataset Notes

### Icon645
ICON645 is a large-scale icon classification dataset with 377 classes and ~645,000 images. A random subset of 40,000 images was evaluated but excluded from the final configuration due to heterogeneous visual style and weak semantic correspondence to EasyRead pictograms.

### Unified Metadata Structure
All datasets are merged into a unified directory. Each image is associated with:
- A single JSON metadata entry (filename, dataset label, prompt info)
- A parallel CSV file for external tool compatibility
- BLIP-generated captions under keys `prompt` and `prompt_model`
- Dataset identifiers for balanced sampling
- EasyRead-related metrics

---

## Appendix B: Validation Dataset (55 Prompts)

| Scene Description | Background | Skin | Hair |
|---|---|---|---|
| The skyline of NYC at night | Red | White | Blonde |
| A person watering a houseplant | Green | Black | Brown |
| A child brushing their teeth | Blue | Asian | Dark Brown |
| A doctor listening to heartbeat | Yellow | Mulatto | Gray |
| A firefighter climbing a ladder | Black | Aztec | Dark Gray |
| A teacher writing on a chalkboard | White | White | Red |
| A bicycle leaning against a wall | Red | Black | Black |
| A dog catching a frisbee | Green | Asian | Blonde |
| An airplane taking off | Blue | Mulatto | Brown |
| A person cooking soup | Yellow | Aztec | Dark Brown |
| A person reading a map outdoors | Black | White | Gray |
| A mom helping child with homework | White | Black | Dark Gray |
| A swimmer diving into a pool | Red | Asian | Red |
| A person running in the rain | Green | Mulatto | Black |
| A chef preparing vegetables | Blue | Aztec | Blonde |
| A person repairing a bicycle tire | Yellow | White | Brown |
| A commuter waiting for a bus | Black | Black | Dark Brown |
| A dentist examining a patient | White | Asian | Gray |
| A person opening a window | Red | Mulatto | Dark Gray |
| A delivery worker carrying package | Green | Aztec | Red |
| A person folding laundry | Blue | White | Black |
| A person feeding a baby | Yellow | Black | Blonde |
| A person practicing yoga | Black | Asian | Brown |
| A police officer directing traffic | White | Mulatto | Dark Brown |
| A tree with falling leaves | Red | Aztec | Gray |
| A cup of coffee on a table | Green | White | Dark Gray |
| A car driving on a mountain road | Blue | Black | Red |
| A lighthouse overlooking the ocean | Yellow | Asian | Black |
| A bus picking up passengers | Black | Mulatto | Blonde |
| A person taking out the trash | White | Aztec | Brown |
| A person tying their shoes | Red | White | Dark Brown |
| A construction worker laying bricks | Green | Black | Gray |
| A nurse giving an injection | Blue | Asian | Dark Gray |
| A person using a laptop | Yellow | Mulatto | Red |
| A person painting a wall | Black | Aztec | Black |
| A person holding an umbrella | White | White | Blonde |
| A bird flying above a lake | Red | Black | Brown |
| A soccer player kicking a ball | Green | Asian | Dark Brown |
| A person waiting at a crosswalk | Blue | Mulatto | Gray |
| A person fishing from a boat | Yellow | Aztec | Dark Gray |
| A person giving a presentation | Black | White | Red |
| A person carrying groceries | White | Black | Black |
| A cat sleeping on a cushion | Red | Asian | Blonde |
| A train arriving at a station | Green | Mulatto | Brown |
| A person washing their hands | Blue | Aztec | Dark Brown |
| A person climbing stairs | Yellow | White | Gray |
| A person chopping vegetables | Black | Black | Dark Gray |
| A person taking a photograph | White | Asian | Red |
| A street market with food stands | Red | Mulatto | Black |
| A person putting on a jacket | Green | Aztec | Blonde |
| A person calling for help | Blue | White | Brown |
| A person holding a first aid kit | Yellow | Black | Dark Brown |
| A person turning off a light switch | Black | Asian | Gray |
| A person cleaning a window | White | Mulatto | Dark Gray |
| A mom reading to a happy child | Red | Black | Blonde |

---

## Appendix C: Experiment Configurations Tested

- Training solely on ARASAAC, with and without controllability attributes
- Training on combined datasets with and without BLIP-generated custom prompts
- Textual inversion to encode EasyRead style into a learnable token (weak results)
- img2img controllability setup to refine base images into EasyRead aesthetic (inconsistent results)

**Best configuration:** All datasets except Icon645, using BLIP-generated prompts and ARASAAC controllability tags.