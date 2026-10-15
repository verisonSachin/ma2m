"""Manga text bubble and SFX removal using LaMa deep inpainting with OpenCV fallback."""

import os
from typing import Union, Optional
import numpy as np
import cv2
import torch
from PIL import Image


class MangaInpainter:
    """Removes text bubbles, caption boxes, and SFX masks from manga art."""

    def __init__(
        self,
        engine: str = "lama",
        model_path: Optional[str] = "models/big-lama.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dilate_mask_pixels: int = 5,
    ):
        self.engine = engine.lower()
        self.model_path = model_path
        self.device = device
        self.dilate_mask_pixels = dilate_mask_pixels
        self.lama_model = None

        if self.engine == "lama" and self.model_path and os.path.exists(self.model_path):
            self._load_lama_model()

    def _load_lama_model(self):
        """Attempt loading TorchScript or state-dict LaMa model."""
        try:
            print(f"[MangaInpainter] Loading LaMa model from {self.model_path} onto {self.device}...")
            self.lama_model = torch.jit.load(self.model_path, map_location=self.device)
            self.lama_model.eval()
        except Exception as e:
            print(f"[MangaInpainter] Could not load TorchScript LaMa model: {e}. Falling back to OpenCV inpainter.")
            self.lama_model = None

    def inpaint(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        mask_input: Union[str, np.ndarray, Image.Image],
    ) -> np.ndarray:
        """
        Inpaint masked regions from the given image.

        Args:
            image_input: RGB image (H, W, 3).
            mask_input: Binary mask (H, W) where non-zero indicates the inpaint area.

        Returns:
            inpainted_rgb: Clean RGB uint8 numpy array with text bubbles removed.
        """
        rgb_img = self._load_rgb(image_input)
        mask = self._load_mask(mask_input, rgb_img.shape[:2])

        # If mask is empty, nothing to inpaint
        if np.count_nonzero(mask) == 0:
            return rgb_img.copy()

        # Dilate mask slightly to prevent bubble rim ghosting
        if self.dilate_mask_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilate_mask_pixels * 2 + 1, self.dilate_mask_pixels * 2 + 1),
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Use LaMa deep model if initialized
        if self.lama_model is not None:
            try:
                return self._inpaint_lama(rgb_img, mask)
            except Exception as e:
                print(f"[MangaInpainter] LaMa inference failed ({e}), falling back to OpenCV Telea.")

        # Robust OpenCV Inpainting Fallback
        return self._inpaint_opencv(rgb_img, mask)

    def _inpaint_lama(self, rgb_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using loaded LaMa neural network."""
        h, w = rgb_img.shape[:2]
        # Pad dimensions to multiples of 8 for UNet compatibility
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8

        padded_img = cv2.copyMakeBorder(rgb_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        padded_mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(padded_mask).unsqueeze(0).float() / 255.0

        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.lama_model(img_tensor, mask_tensor)
            if isinstance(output_tensor, dict):
                output_tensor = output_tensor.get("predicted_image", output_tensor.get("inpainted"))
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
            res_np = (output_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        # Unpad back to original dimensions
        clean_rgb = res_np[:h, :w]

        # Blend with original on unmasked regions
        binary_mask_3c = np.repeat((mask > 0)[:, :, np.newaxis], 3, axis=2)
        blended = np.where(binary_mask_3c, clean_rgb, rgb_img)
        return blended

    def _inpaint_opencv(self, rgb_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Multi-stage OpenCV Navier-Stokes + Fast Marching Method for clean manga lines."""
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # Telea fast marching algorithm handles smooth gradients and manga halftones well
        inpainted_bgr = cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # Post-filter to smooth tone transitions inside inpainted region
        smoothed = cv2.bilateralFilter(inpainted_bgr, 5, 40, 40)
        mask_3c = np.repeat((mask > 0)[:, :, np.newaxis], 3, axis=2)
        blended_bgr = np.where(mask_3c, smoothed, bgr)

        return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    def _load_rgb(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise ValueError(f"Cannot read image: {image_input}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    def _load_mask(self, mask_input: Union[str, np.ndarray, Image.Image], target_shape: tuple) -> np.ndarray:
        if isinstance(mask_input, str):
            mask = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
        elif isinstance(mask_input, Image.Image):
            mask = np.array(mask_input.convert("L"))
        elif isinstance(mask_input, np.ndarray):
            if mask_input.ndim == 3:
                mask = cv2.cvtColor(mask_input, cv2.COLOR_RGB2GRAY)
            else:
                mask = mask_input
        else:
            raise TypeError(f"Unsupported mask type: {type(mask_input)}")

        if mask.shape[:2] != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return bin_mask
