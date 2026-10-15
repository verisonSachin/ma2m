"""Reference color palette management, K-Means palette extraction, and cross-panel color harmonization."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans


@dataclass
class CharacterPalette:
    """Stores key chromatic anchors for a character."""
    name: str = "protagonist"
    hair: str = "#C0392B"
    skin: str = "#F5CBA7"
    eyes: str = "#2980B9"
    primary_clothing: str = "#1B2631"
    secondary_clothing: str = "#E74C3C"
    accent: Optional[str] = "#F39C12"
    reference_image_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_prompt_tags(self) -> str:
        """Translate palette HEX anchors into descriptive prompt tags."""
        tags = []
        tags.append(f"{self._hex_to_color_name(self.hair)} hair")
        tags.append(f"{self._hex_to_color_name(self.skin)} skin")
        tags.append(f"{self._hex_to_color_name(self.eyes)} eyes")
        tags.append(f"{self._hex_to_color_name(self.primary_clothing)} outfit")
        return ", ".join(tags)

    @staticmethod
    def _hex_to_color_name(hex_code: str) -> str:
        """Approximates simple color names for diffusion prompting."""
        r, g, b = PaletteManager.hex_to_rgb(hex_code)
        if r > 180 and g > 180 and b > 180:
            return "white"
        if r < 50 and g < 50 and b < 50:
            return "black"
        if r > 150 and g > 130 and b < 100:
            return "blonde"
        if r > 160 and g < 70 and b < 70:
            return "crimson red"
        if r > 180 and g < 120 and b < 60:
            return "orange"
        if r < 70 and g < 100 and b > 150:
            return "blue"
        if r < 70 and g > 130 and b < 100:
            return "green"
        if r > 130 and g < 70 and b > 130:
            return "purple"
        if r > 200 and g > 160 and b > 140:
            return "fair peach"
        return "vibrant"


class PaletteManager:
    """Harmonizes color across panels and extracts palettes from reference images."""

    def __init__(self, default_palette: Optional[CharacterPalette] = None):
        self.active_palette = default_palette or CharacterPalette()
        self.reference_features: Optional[np.ndarray] = None

    @staticmethod
    def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        """Convert '#RRGGBB' to (R, G, B) tuple."""
        clean = hex_str.strip().lstrip("#")
        if len(clean) == 3:
            clean = "".join(c * 2 for c in clean)
        return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)

    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert (R, G, B) tuple to '#RRGGBB'."""
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

    def extract_palette_from_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        k: int = 5,
    ) -> List[str]:
        """
        Extract k dominant colors from reference artwork using K-Means clustering.
        Returns list of HEX color codes.
        """
        rgb_img = self._load_rgb(image_input)
        # Downsample for fast clustering
        h, w = rgb_img.shape[:2]
        small = cv2.resize(rgb_img, (128, 128), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3).astype(np.float32)

        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_.astype(int)

        # Sort centers by luminance
        centers = sorted(centers, key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
        hex_colors = [self.rgb_to_hex(tuple(c)) for c in centers]
        return hex_colors

    def harmonize_panel(
        self,
        panel_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        strength: float = 0.85,
    ) -> np.ndarray:
        """
        Applies Reinhard LAB color transfer + histogram matching to lock color consistency.
        Transfers chromatic mean and standard deviation from reference artwork.
        """
        if panel_rgb.shape[:2] != reference_rgb.shape[:2]:
            ref_resized = cv2.resize(
                reference_rgb,
                (panel_rgb.shape[1], panel_rgb.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            ref_resized = reference_rgb

        # Convert to LAB space
        src_lab = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_resized, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Split channels
        l_s, a_s, b_s = cv2.split(src_lab)
        l_r, a_r, b_r = cv2.split(ref_lab)

        # Compute stats
        mean_l_s, std_l_s = np.mean(l_s), np.std(l_s) + 1e-5
        mean_a_s, std_a_s = np.mean(a_s), np.std(a_s) + 1e-5
        mean_b_s, std_b_s = np.mean(b_s), np.std(b_s) + 1e-5

        mean_l_r, std_l_r = np.mean(l_r), np.std(l_r) + 1e-5
        mean_a_r, std_a_r = np.mean(a_r), np.std(a_r) + 1e-5
        mean_b_r, std_b_r = np.mean(b_r), np.std(b_r) + 1e-5

        # Transfer channels
        l_trans = ((l_s - mean_l_s) * (std_l_r / std_l_s)) + mean_l_r
        a_trans = ((a_s - mean_a_s) * (std_a_r / std_a_s)) + mean_a_r
        b_trans = ((b_s - mean_b_s) * (std_b_r / std_b_s)) + mean_b_r

        # Clip values
        l_trans = np.clip(l_trans, 0, 255)
        a_trans = np.clip(a_trans, 0, 255)
        b_trans = np.clip(b_trans, 0, 255)

        merged_lab = cv2.merge([l_trans, a_trans, b_trans]).astype(np.uint8)
        transferred_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

        # Blend with original by strength
        result = np.clip(
            (1.0 - strength) * panel_rgb.astype(np.float32) + strength * transferred_rgb.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        return result

    def compute_delta_e(self, color1_rgb: Tuple[int, int, int], color2_rgb: Tuple[int, int, int]) -> float:
        """
        Compute CIE76 Delta-E chromatic variance between two RGB colors.
        Returns distance score (typically < 2.3 is imperceptible, < 5.0 is acceptable consistency).
        """
        c1 = np.uint8([[list(color1_rgb)]])
        c2 = np.uint8([[list(color2_rgb)]])

        lab1 = cv2.cvtColor(c1, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
        lab2 = cv2.cvtColor(c2, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]

        delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2))
        return float(delta_e)

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
