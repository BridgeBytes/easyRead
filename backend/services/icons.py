"""
Generate EasyRead-style icons from text prompts using the fine-tuned LoRA model.

Usage:
    from generate_icon import IconGenerator

    generator = IconGenerator()
    image = generator.generate("a person reading a book")
    image.save("output.png")
"""

import torch
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image
from logging import getLogger

logger = getLogger(__name__)


class IconGenerator:
    """Generate EasyRead-style pictogram icons from text prompts."""

    DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
    DEFAULT_NEGATIVE_PROMPT = "blurry, photo, photograph, realistic, complex, detailed background"

    def __init__(
        self,
        lora_weights_path: Optional[str] = None,
        base_model: str = DEFAULT_BASE_MODEL,
        device: Optional[str] = None,
        instance_token: str = "sks"
    ):
        """
        Initialize the icon generator.

        Args:
            lora_weights_path: Path to LoRA weights directory. If None, uses the
                               default model/ directory in this project.
            base_model: Base Stable Diffusion model to use.
            device: Device to run on ('cuda', 'mps', or 'cpu'). Auto-detected if None.
            instance_token: Token that triggers the EasyRead style (default: 'sks').
        """
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        from peft import PeftModel

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                logger.info("CUDA is available. Using CUDA device.")
                device = "cuda"
            elif torch.backends.mps.is_available():
                logger.info("MPS is available. Using MPS device.")
                device = "mps"
            else:
                logger.info("Using CPU device.")
                device = "cpu"

        self.device = device
        self.instance_token = instance_token

        # Default to model/ in project root
        if lora_weights_path is None:
            lora_weights_path = Path(__file__).parent.parent / "weights"

        print(f"Loading base model: {base_model}")
        print(f"Device: {device}")

        # Load pipeline with appropriate dtype
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            safety_checker=None
        )

        # Use faster scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

        # Load LoRA weights
        if lora_weights_path and Path(lora_weights_path).exists():
            print(f"Loading LoRA weights from: {lora_weights_path}")
            self.pipeline.unet = PeftModel.from_pretrained(
                self.pipeline.unet,
                lora_weights_path
            )
        else:
            print("No LoRA weights found, using base model")

        self.pipeline = self.pipeline.to(device)
        self.pipeline.enable_attention_slicing()

        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        size: int = 256,
        seed: Optional[int] = None,
        use_instance_token: bool = True,
        background_color: Optional[str] = None,
        skin_color: Optional[str] = None,
        hair_color: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a single icon from a text prompt.

        Args:
            prompt: Text description of the icon to generate.
                    Examples: "person reading a book", "dog running", "tree"
            negative_prompt: What to avoid in generation. Uses default if None.
            num_inference_steps: Number of denoising steps (more = higher quality but slower).
            guidance_scale: How closely to follow the prompt (5-10 typical).
            size: Image size in pixels (256 recommended for this model).
            seed: Random seed for reproducibility.
            use_instance_token: Whether to prepend the instance token (enables EasyRead style).
            background_color: Optional background color (red, green, blue, yellow, black, white).
            skin_color: Optional skin tone (white, black, asian, mulatto, aztec).
            hair_color: Optional hair color (blonde, brown, darkBrown, gray, red, black).

        Returns:
            PIL Image of the generated icon.
        """
        # Build the full prompt
        full_prompt = prompt

        # Add color attributes if specified
        color_parts = []
        if background_color:
            color_parts.append(f"background color: {background_color}")
        if skin_color:
            color_parts.append(f"skin color: {skin_color}")
        if hair_color:
            color_parts.append(f"hair color: {hair_color}")

        if color_parts:
            full_prompt = f"{full_prompt}; {'; '.join(color_parts)}"

        # Prepend instance token
        if use_instance_token and self.instance_token:
            full_prompt = f"{self.instance_token} {full_prompt}"

        # Use default negative prompt
        if negative_prompt is None:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"Generating: {full_prompt}")

        with torch.inference_mode():
            result = self.pipeline(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=size,
                width=size,
                generator=generator
            )

        return result.images[0]

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate multiple icons from a list of prompts.

        Args:
            prompts: List of text descriptions.
            **kwargs: Additional arguments passed to generate().

        Returns:
            List of PIL Images.
        """
        images = []
        for i, prompt in enumerate(prompts):
            print(f"\nGenerating {i+1}/{len(prompts)}...")
            seed = kwargs.get('seed')
            if seed is not None:
                kwargs['seed'] = seed + i
            images.append(self.generate(prompt, **kwargs))
        return images


def generate_icon(
    prompt: str,
    output_path: Optional[str] = None,
    **kwargs
) -> Image.Image:
    """
    Convenience function to generate a single icon.

    Args:
        prompt: Text description of the icon.
        output_path: Optional path to save the image.
        **kwargs: Additional arguments passed to IconGenerator.generate().

    Returns:
        PIL Image of the generated icon.
    """
    generator = IconGenerator()
    image = generator.generate(prompt, **kwargs)

    if output_path:
        image.save(output_path)
        print(f"Saved to: {output_path}")

    return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate EasyRead-style icons from text prompts")
    parser.add_argument("prompt", type=str, help="Text description of the icon to generate")
    parser.add_argument("-o", "--output", type=str, default="generated_icon.png", help="Output filename")
    parser.add_argument("-n", "--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--size", type=int, default=256, help="Image size in pixels")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--background", type=str, default=None, help="Background color")
    parser.add_argument("--skin", type=str, default=None, help="Skin color")
    parser.add_argument("--hair", type=str, default=None, help="Hair color")
    parser.add_argument("--no-instance-token", action="store_true", help="Don't use instance token")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--lora-weights", type=str, default=None, help="Path to LoRA weights")

    args = parser.parse_args()

    # Initialize generator
    generator = IconGenerator(
        lora_weights_path=args.lora_weights,
        device=args.device
    )

    # Generate images
    for i in range(args.num_images):
        seed = args.seed + i if args.seed is not None else None

        image = generator.generate(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            size=args.size,
            seed=seed,
            use_instance_token=not args.no_instance_token,
            background_color=args.background,
            skin_color=args.skin,
            hair_color=args.hair
        )

        # Save with numbered suffix if generating multiple
        if args.num_images == 1:
            output_path = args.output
        else:
            stem = Path(args.output).stem
            suffix = Path(args.output).suffix
            output_path = f"{stem}_{i+1}{suffix}"

        image.save(output_path)
        print(f"Saved: {output_path}")

    print("\nDone!")
