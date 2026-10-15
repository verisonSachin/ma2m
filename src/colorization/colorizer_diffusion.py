"""Diffusion-driven colorization using SDXL/SD1.5 with ControlNet Lineart Anime & IP-Adapter."""

import os
from typing import Optional, Union, Tuple
import numpy as np
import cv2
import torch
from PIL import Image

from src.colorization.palette_manager import CharacterPalette, PaletteManager
from src.colorization.lineart_cleaner import LineartCleaner
from src.colorization.free_api_client import FreeImageAPIClient


class MangaColorizer:
    """Colorizes black & white manga lineart with reference palette locking."""

    def __init__(
        self,
        base_model: str = "cagliostrolab/animagine-xl-3.1",
        controlnet_model: str = "lllyasviel/control_v11p_sd15_lineart_anime",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_ip_adapter: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        use_api: Optional[bool] = None,
    ):
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.device = device
        self.use_ip_adapter = use_ip_adapter
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        # API mode configuration
        env_use_api = os.getenv("USE_IMAGE_API", "false").lower() in ("true", "1", "yes")
        self.use_api = use_api if use_api is not None else env_use_api
        self.api_client = FreeImageAPIClient()

        self.pipeline = None
        self.lineart_cleaner = LineartCleaner()
        self.palette_manager = PaletteManager()

        # Initialize pipeline lazily or on demand if not in API-only mode
        if not self.use_api:
            self._init_diffusers_pipeline()

    def _init_diffusers_pipeline(self):
        """Attempts to load ControlNet lineart pipeline if dependencies and GPU are present."""
        if not torch.cuda.is_available():
            print("[MangaColorizer] CUDA unavailable. Using free API client or procedural engine.")
            return

        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetPipeline,
                UniPCMultistepScheduler,
            )

            print(f"[MangaColorizer] Loading ControlNet lineart model: {self.controlnet_model}...")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            print(f"[MangaColorizer] Initializing Stable Diffusion pipeline...")
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
            )
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.to(self.device)
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
            print("[MangaColorizer] Diffusion colorization pipeline successfully loaded.")
        except Exception as e:
            print(f"[MangaColorizer] Could not initialize diffusion models ({e}). Using API / procedural fallback.")
            self.pipeline = None

    def colorize(
        self,
        panel_input: Union[str, np.ndarray, Image.Image],
        palette: Optional[CharacterPalette] = None,
        custom_prompt: Optional[str] = None,
        reference_image: Optional[Union[str, np.ndarray, Image.Image]] = None,
        target_size: Tuple[int, int] = (768, 512),  # (H, W)
    ) -> np.ndarray:
        """
        Colorize a manga panel while locking lineart and chromatic reference.

        Returns:
            colorized_rgb: RGB uint8 numpy array (H, W, 3).
        """
        raw_rgb = self._load_rgb(panel_input)
        smoothed_base, lineart_rgb = self.lineart_cleaner.clean(raw_rgb)

        active_pal = palette or self.palette_manager.active_palette

        # Build prompt incorporating palette color descriptors
        palette_tags = active_pal.to_prompt_tags()
        base_prompt = "masterpiece, best quality, anime aesthetic, vibrant studio lighting, crisp colors"
        if custom_prompt:
            prompt = f"{base_prompt}, {custom_prompt}, {palette_tags}"
        else:
            prompt = f"{base_prompt}, {palette_tags}"

        neg_prompt = "monochrome, grayscale, b&w, lowres, bad quality, oversaturated, deformed, blurry"

        # Resize lineart for model input
        h_out, w_out = target_size
        lineart_resized = cv2.resize(lineart_rgb, (w_out, h_out), interpolation=cv2.INTER_AREA)

        colorized_np = None

        # 1. Option: Free/Open-Source Image API (Pollinations, Hugging Face, or Docker local API)
        if self.use_api or os.getenv("ENABLE_FREE_API_FALLBACK", "true").lower() in ("true", "1"):
            api_result = self.api_client.generate_or_colorize(
                prompt=prompt,
                lineart_rgb=lineart_resized,
                negative_prompt=neg_prompt,
                width=w_out,
                height=h_out,
            )
            if api_result is not None:
                # Composite API color with sharp original lineart
                colorized_np = self._blend_with_lineart(api_result, lineart_resized)

        # 2. Option: Local GPU Diffusers Pipeline
        if colorized_np is None and self.pipeline is not None:
            try:
                lineart_pil = Image.fromarray(lineart_resized)
                generator = torch.Generator(device=self.device).manual_seed(42)

                result_pil = self.pipeline(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    image=lineart_pil,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                ).images[0]

                colorized_np = np.array(result_pil.convert("RGB"))
            except Exception as e:
                print(f"[MangaColorizer] Local diffusion step failed ({e}). Reverting to procedural anime colorizer.")
                colorized_np = None

        # 3. Option: High-Fidelity Procedural Anime Cel-Shading Fallback
        if colorized_np is None:
            colorized_np = self._procedural_anime_colorizer(smoothed_base, lineart_resized, active_pal)

        # Cross-reference harmonization if reference image provided
        if reference_image is not None:
            ref_rgb = self._load_rgb(reference_image)
            colorized_np = self.palette_manager.harmonize_panel(colorized_np, ref_rgb, strength=0.6)

        return colorized_np

    def _blend_with_lineart(self, color_rgb: np.ndarray, lineart_rgb: np.ndarray) -> np.ndarray:
        """Blend generated color with manga lineart to prevent ink loss."""
        if color_rgb.shape[:2] != lineart_rgb.shape[:2]:
            color_rgb = cv2.resize(color_rgb, (lineart_rgb.shape[1], lineart_rgb.shape[0]))
        line_norm = cv2.cvtColor(lineart_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        line_mult = np.repeat(line_norm[:, :, np.newaxis], 3, axis=2)
        blended = color_rgb.astype(np.float32) * (0.2 + 0.8 * line_mult)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _procedural_anime_colorizer(
        self,
        base_gray_rgb: np.ndarray,
        lineart_rgb: np.ndarray,
        palette: CharacterPalette,
    ) -> np.ndarray:
        """
        Fast, high-fidelity procedural anime flat/cel-shading colorizer.
        Used as standalone engine or fallback when GPU/heavy diffusion weights are absent.
        """
        h, w = lineart_rgb.shape[:2]
        base_resized = cv2.resize(base_gray_rgb, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(base_resized, cv2.COLOR_RGB2GRAY)

        # Segment luminance bands for cel shading (highlights, midtones, shadows)
        skin_rgb = np.array(PaletteManager.hex_to_rgb(palette.skin), dtype=np.float32)
        hair_rgb = np.array(PaletteManager.hex_to_rgb(palette.hair), dtype=np.float32)
        cloth_rgb = np.array(PaletteManager.hex_to_rgb(palette.primary_clothing), dtype=np.float32)

        tiers = np.digitize(gray, [70, 140, 210])
        color_canvas = np.zeros((h, w, 3), dtype=np.float32)

        # Tier 0 (Deep shadow / clothing / hair core)
        mask0 = (tiers == 0)[:, :, np.newaxis]
        color_canvas += mask0 * (cloth_rgb * 0.75)

        # Tier 1 (Midtones: secondary clothing / shadow skin)
        mask1 = (tiers == 1)[:, :, np.newaxis]
        color_canvas += mask1 * (hair_rgb * 0.9)

        # Tier 2 (Bright midtones: character skin / base illumination)
        mask2 = (tiers == 2)[:, :, np.newaxis]
        color_canvas += mask2 * (skin_rgb * 0.95)

        # Tier 3 (Highlights: rim lighting / environment light)
        mask3 = (tiers == 3)[:, :, np.newaxis]
        highlight_color = np.array([250, 245, 235], dtype=np.float32)
        color_canvas += mask3 * highlight_color

        line_norm = cv2.cvtColor(lineart_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        line_mult = np.repeat(line_norm[:, :, np.newaxis], 3, axis=2)

        final_color = color_canvas * (0.15 + 0.85 * line_mult)
        final_color = np.clip(final_color, 0, 255).astype(np.uint8)

        final_smooth = cv2.bilateralFilter(final_color, 5, 50, 50)
        return final_smooth

    def _load_rgb(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise ValueError(f"Could not load image: {image_input}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input
        raise TypeError(f"Unsupported image input type: {type(image_input)}")
