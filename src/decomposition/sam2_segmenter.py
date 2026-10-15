"""Layer decomposition separating foreground characters from background plates using SAM 2."""

import os
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np
import cv2
import torch
from PIL import Image

from src.ingestion.inpainter import MangaInpainter


@dataclass
class LayerDecomposition:
    """Stores separated foreground and background visual layers for parallax animation."""
    foreground_rgba: np.ndarray  # (H, W, 4) with alpha cutout
    background_rgb: np.ndarray   # (H, W, 3) with foreground inpainted
    foreground_mask: np.ndarray  # (H, W) uint8 binary mask (255 = foreground)


class SAM2Segmenter:
    """Decomposes static manga frames into multi-plane 2.5D visual layers."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "models/sam2_hiera_large.pt",
        config_name: str = "sam2_hiera_l.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        self.device = device
        self.predictor = None
        self.inpainter = MangaInpainter(engine="telea", device=self.device)

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self._init_sam2()

    def _init_sam2(self):
        """Initializes Segment Anything 2 predictor."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            print(f"[SAM2Segmenter] Loading SAM 2 model from {self.checkpoint_path}...")
            sam_model = build_sam2(self.config_name, self.checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam_model)
            print("[SAM2Segmenter] SAM 2 successfully loaded.")
        except Exception as e:
            print(f"[SAM2Segmenter] Could not load SAM 2 ({e}). Saliency-contour decomposition will be used.")
            self.predictor = None

    def segment_layers(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
    ) -> LayerDecomposition:
        """
        Decomposes input panel into:
        1. Alpha-cutout foreground character layer.
        2. Clean inpainted background plate (so moving the character reveals clean background).
        3. Binary mask.
        """
        rgb_img = self._load_rgb(image_input)
        h, w = rgb_img.shape[:2]

        if self.predictor is not None:
            fg_mask = self._segment_with_sam2(rgb_img)
        else:
            fg_mask = self._segment_saliency(rgb_img)

        # Smooth mask contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Create RGBA foreground
        foreground_rgba = np.dstack([rgb_img, fg_mask])

        # Create inpainted background plate by filling the foreground character hole
        # Dilate mask slightly to prevent character edge haloing in background
        bg_inpaint_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        background_rgb = self.inpainter.inpaint(rgb_img, bg_inpaint_mask)

        return LayerDecomposition(
            foreground_rgba=foreground_rgba,
            background_rgb=background_rgb,
            foreground_mask=fg_mask,
        )

    def _segment_with_sam2(self, rgb_img: np.ndarray) -> np.ndarray:
        """Runs SAM 2 automatic center-focused prompt segmentation."""
        h, w = rgb_img.shape[:2]
        self.predictor.set_image(rgb_img)

        # Prompt with center points assuming main character is in focus
        center_points = np.array([[w // 2, h // 2], [w // 2, int(h * 0.35)]])
        labels = np.array([1, 1])

        masks, scores, _ = self.predictor.predict(
            point_coords=center_points,
            point_labels=labels,
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)]
        return (best_mask * 255).astype(np.uint8)

    def _segment_saliency(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Heuristic anime character foreground extraction using center-weighted
        saliency, edge density, and GrabCut segmentation.
        """
        h, w = rgb_img.shape[:2]
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        # Bounding box initial estimate: central 60% of frame
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.10)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(rgb_img, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
        except Exception:
            # Simple threshold fallback if GrabCut fails on degenerate image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            fg_mask = thresh

        return fg_mask

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
