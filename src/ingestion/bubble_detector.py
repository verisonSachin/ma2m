"""Text bubble and SFX (sound effects) detection and mask generation for manga panels."""

import os
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import numpy as np
import cv2
from PIL import Image


@dataclass
class BubbleRegion:
    """Represents an identified speech bubble or SFX region."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    bubble_type: str  # "speech", "sfx", "thought"
    polygon: Optional[List[Tuple[int, int]]] = None


class BubbleDetector:
    """Detects dialog balloons, caption boxes, and prominent SFX in manga panels."""

    def __init__(
        self,
        white_threshold: int = 220,
        min_bubble_area: int = 350,
        max_bubble_ratio: float = 0.65,
        edge_dilation_kernel: int = 7,
    ):
        self.white_threshold = white_threshold
        self.min_bubble_area = min_bubble_area
        self.max_bubble_ratio = max_bubble_ratio
        self.edge_dilation_kernel = edge_dilation_kernel

    def detect(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
    ) -> Tuple[np.ndarray, List[BubbleRegion]]:
        """
        Analyze a manga panel and generate an inpainting mask and bubble regions.

        Returns:
            mask: Binary uint8 numpy array (H, W) where 255 indicates regions to inpaint.
            regions: List of BubbleRegion objects.
        """
        img_rgb = self._load_rgb(image_input)
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        total_panel_area = float(h * w)
        bubble_mask = np.zeros((h, w), dtype=np.uint8)
        regions: List[BubbleRegion] = []

        # 1. White / Light background segmentation (typical for manga speech bubbles)
        # Manga bubbles typically have high luminance interior surrounded by dark borders
        _, thresh_white = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)

        # 2. Text density detection within candidates
        # Find high-gradient text stroke regions inside bubbles
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)
        grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
        _, text_edges = cv2.threshold(grad_mag, 60, 255, cv2.THRESH_BINARY)

        # 3. Find connected components on white regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh_white, connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_bubble_area or area > (total_panel_area * self.max_bubble_ratio):
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # Aspect ratio check
            aspect = bw / float(bh + 1e-5)
            if aspect < 0.15 or aspect > 6.0:
                continue

            # Candidate mask for this component
            comp_mask = (labels == i).astype(np.uint8) * 255

            # Analyze internal text presence
            internal_text = cv2.bitwise_and(text_edges, text_edges, mask=comp_mask)
            text_pixel_ratio = np.count_nonzero(internal_text) / float(area + 1e-5)

            # Check border convexity / roundness
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area + 1e-5)

            # Speech bubbles are generally solid convex/oval shapes with internal text/strokes
            if (solidity > 0.65 and text_pixel_ratio > 0.015) or (text_pixel_ratio > 0.05):
                # Valid speech bubble or SFX block
                bubble_mask = cv2.bitwise_or(bubble_mask, comp_mask)

                # Classify type
                btype = "sfx" if solidity < 0.75 else "speech"
                conf = min(1.0, max(0.5, float(solidity * 0.7 + text_pixel_ratio * 3.0)))

                regions.append(
                    BubbleRegion(
                        bbox=(x, y, x + bw, y + bh),
                        confidence=conf,
                        bubble_type=btype,
                        polygon=[(int(pt[0][0]), int(pt[0][1])) for pt in cnt],
                    )
                )

        # 4. Dilate the mask to encompass bubble outer stroke borders and tails
        if self.edge_dilation_kernel > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.edge_dilation_kernel, self.edge_dilation_kernel),
            )
            bubble_mask = cv2.dilate(bubble_mask, k, iterations=2)

        return bubble_mask, regions

    def _load_rgb(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Could not load image at {image_input}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input
        else:
            raise TypeError(f"Invalid input type: {type(image_input)}")
