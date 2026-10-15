"""Manga panel segmentation and Japanese Right-to-Left (RTL) reading order traversal."""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image


@dataclass
class Panel:
    """Represents a segmented manga panel."""
    page_idx: int
    panel_idx: int
    global_order: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    image: np.ndarray  # RGB numpy array (H, W, 3)
    polygon: Optional[List[Tuple[int, int]]] = None
    confidence: float = 1.0
    label: str = "panel"
    metadata: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )


class PanelExtractor:
    """Detects manga panels from raw page scans and sequences them in RTL order."""

    def __init__(
        self,
        model_path: Optional[str] = "models/yolov8_manga.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        min_panel_area_ratio: float = 0.02,
        fallback_contour_method: bool = True,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_panel_area_ratio = min_panel_area_ratio
        self.fallback_contour_method = fallback_contour_method
        self.yolo_model = None

        if self.model_path and os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.model_path)
            except Exception as e:
                print(f"[PanelExtractor] Failed to load YOLOv8 model from {self.model_path}: {e}. Using contour fallback.")
                self.yolo_model = None

    def extract_panels_from_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        page_idx: int = 0,
    ) -> List[Panel]:
        """Extract panels from a single manga page image."""
        rgb_img = self._load_image_rgb(image_input)
        h, w = rgb_img.shape[:2]

        raw_boxes = []
        if self.yolo_model is not None:
            raw_boxes = self._detect_yolo(rgb_img)

        # Fallback to contour detection if YOLO found nothing or model is absent
        if len(raw_boxes) == 0 and self.fallback_contour_method:
            raw_boxes = self._detect_contours(rgb_img)

        # If still no panels detected, treat the entire page as a single panel
        if len(raw_boxes) == 0:
            raw_boxes = [(0, 0, w, h, 1.0)]

        # Filter out tiny boxes or full-page wrappers if smaller panels exist
        valid_boxes = self._filter_panels(raw_boxes, w, h)

        # Order panels according to Japanese manga reading flow (RTL, Top-to-Bottom)
        ordered_boxes = self._sort_panels_rtl(valid_boxes)

        panels = []
        for panel_idx, (x1, y1, x2, y2, conf) in enumerate(ordered_boxes):
            crop = rgb_img[y1:y2, x1:x2].copy()
            panel = Panel(
                page_idx=page_idx,
                panel_idx=panel_idx,
                global_order=panel_idx,
                bbox=(x1, y1, x2, y2),
                image=crop,
                confidence=float(conf),
            )
            panels.append(panel)

        return panels

    def process_pages(
        self,
        page_inputs: List[Union[str, np.ndarray, Image.Image]],
    ) -> List[Panel]:
        """Process a multi-page manga sequence and assign global reading order."""
        all_panels: List[Panel] = []
        global_counter = 0

        for page_idx, page_input in enumerate(page_inputs):
            page_panels = self.extract_panels_from_image(page_input, page_idx=page_idx)
            for p in page_panels:
                p.global_order = global_counter
                global_counter += 1
                all_panels.append(p)

        return all_panels

    def _detect_yolo(self, rgb_img: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Run YOLOv8 inference for panel bounding box detection."""
        results = self.yolo_model.predict(
            rgb_img,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf))
        return boxes

    def _detect_contours(self, rgb_img: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Contour-based panel boundary extraction using adaptive binarization & morphology."""
        h, w = rgb_img.shape[:2]
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        # Bilateral filter to smooth screentones without destroying panel borders
        smoothed = cv2.bilateralFilter(gray, 7, 50, 50)

        # Adaptive thresholding to capture panel gutters and outer border lines
        binary = cv2.adaptiveThreshold(
            smoothed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            10,
        )

        # Morphological horizontal and vertical kernels to connect broken panel borders
        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        horiz = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_horiz)
        vert = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_vert)
        borders = cv2.bitwise_or(horiz, vert)

        # Invert to find panel regions as connected components
        inv_borders = cv2.bitwise_not(borders)
        contours, _ = cv2.findContours(inv_borders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        page_area = float(h * w)
        min_area = page_area * self.min_panel_area_ratio
        boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Exclude full page border artifacts
            if bw >= w * 0.98 and bh >= h * 0.98:
                continue
            # Aspect ratio sanity check (panels are rarely razor thin)
            aspect = bw / float(bh + 1e-5)
            if aspect < 0.08 or aspect > 12.0:
                continue

            # Add box with confidence based on rectangularity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            rect_ratio = area / (bw * bh + 1e-5)
            conf = min(1.0, max(0.4, float(rect_ratio)))
            boxes.append((x, y, x + bw, y + bh, conf))

        return boxes

    def _filter_panels(
        self,
        boxes: List[Tuple[int, int, int, int, float]],
        page_w: int,
        page_h: int,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Clip coordinates to page limits and remove duplicate or embedded boxes."""
        clipped = []
        for x1, y1, x2, y2, conf in boxes:
            cx1 = max(0, min(page_w - 1, x1))
            cy1 = max(0, min(page_h - 1, y1))
            cx2 = max(cx1 + 10, min(page_w, x2))
            cy2 = max(cy1 + 10, min(page_h, y2))
            area = (cx2 - cx1) * (cy2 - cy1)
            if area >= (page_w * page_h * self.min_panel_area_ratio):
                clipped.append((cx1, cy1, cx2, cy2, conf))

        # Non-Maximum Suppression / containment filtering
        keep = []
        for i, box_a in enumerate(clipped):
            xa1, ya1, xa2, ya2, ca = box_a
            area_a = (xa2 - xa1) * (ya2 - ya1)
            is_subsumed = False
            for j, box_b in enumerate(clipped):
                if i == j:
                    continue
                xb1, yb1, xb2, yb2, cb = box_b
                # Check intersection
                inter_x1 = max(xa1, xb1)
                inter_y1 = max(ya1, yb1)
                inter_x2 = min(xa2, xb2)
                inter_y2 = min(ya2, yb2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    # If box_a is heavily contained inside a larger box_b (> 85%), keep the larger or smaller depending on page structure
                    if inter_area / area_a > 0.85 and area_a < (xb2 - xb1) * (yb2 - yb1) * 0.95:
                        # Box a is a subset
                        pass
            keep.append(box_a)

        return keep if keep else clipped

    def _sort_panels_rtl(
        self,
        boxes: List[Tuple[int, int, int, int, float]],
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Sort panels in Japanese Manga reading order:
        1. Vertical tiers from Top to Bottom.
        2. Within each tier, Right to Left (X descending).
        """
        if not boxes:
            return []

        # Sort initially by Y-min
        sorted_by_y = sorted(boxes, key=lambda b: (b[1], -b[2]))

        # Group into horizontal tiers based on vertical overlap
        tiers: List[List[Tuple[int, int, int, int, float]]] = []
        for box in sorted_by_y:
            x1, y1, x2, y2, conf = box
            box_height = y2 - y1
            placed = False

            for tier in tiers:
                # Calculate vertical overlap with existing tier
                tier_y1 = min(b[1] for b in tier)
                tier_y2 = max(b[3] for b in tier)
                tier_height = tier_y2 - tier_y1

                inter_y1 = max(y1, tier_y1)
                inter_y2 = min(y2, tier_y2)
                overlap = max(0, inter_y2 - inter_y1)

                min_h = min(box_height, tier_height)
                if min_h > 0 and (overlap / min_h) > 0.35:
                    tier.append(box)
                    placed = True
                    break

            if not placed:
                tiers.append([box])

        # Sort tiers top to bottom (by average Y)
        tiers.sort(key=lambda tier: np.mean([(b[1] + b[3]) / 2.0 for b in tier]))

        # Within each tier, sort right to left (X center or X2 descending)
        ordered: List[Tuple[int, int, int, int, float]] = []
        for tier in tiers:
            tier_sorted = sorted(tier, key=lambda b: (b[0] + b[2]) / 2.0, reverse=True)
            ordered.extend(tier_sorted)

        return ordered

    def annotate_panels(
        self,
        page_img: Union[str, np.ndarray, Image.Image],
        panels: List[Panel],
    ) -> np.ndarray:
        """Draw bounding boxes and reading order badge labels onto the page image."""
        canvas = self._load_image_rgb(page_img).copy()
        for p in panels:
            x1, y1, x2, y2 = p.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (235, 87, 87), 3)

            # Badge overlay
            badge_text = f"#{p.global_order + 1} (P{p.page_idx + 1}-{p.panel_idx + 1})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, _ = cv2.getTextSize(badge_text, font, font_scale, thickness)

            badge_x = x1 + 8
            badge_y = y1 + text_size[1] + 12
            cv2.rectangle(
                canvas,
                (badge_x - 4, badge_y - text_size[1] - 6),
                (badge_x + text_size[0] + 6, badge_y + 6),
                (33, 37, 41),
                -1,
            )
            cv2.putText(
                canvas,
                badge_text,
                (badge_x, badge_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return canvas

    def _load_image_rgb(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Load image as an RGB uint8 numpy array."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path not found: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Failed to read image at {image_input}")
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
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
