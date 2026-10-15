"""Temporal deflickering and anime lineart restoration across video frames."""

from typing import List
import numpy as np
import cv2


class TemporalDeflicker:
    """Eliminates diffusion luminance flickering while preserving sharp anime lineart."""

    def __init__(
        self,
        temporal_window: int = 5,
        frequency_filter_strength: float = 0.4,
        preserve_lineart_edges: bool = True,
    ):
        self.temporal_window = temporal_window
        self.frequency_filter_strength = frequency_filter_strength
        self.preserve_lineart_edges = preserve_lineart_edges

    def deflicker_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Smooth temporal luminance and color fluctuations across consecutive frames.

        Args:
            frames: List of RGB uint8 video frames.

        Returns:
            clean_frames: List of deflickered RGB uint8 frames.
        """
        n = len(frames)
        if n < 3:
            return frames

        # Convert all frames to LAB color space for separated luminance (L) processing
        lab_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2LAB).astype(np.float32) for f in frames]
        h, w = frames[0].shape[:2]

        smoothed_lab = []
        half_w = self.temporal_window // 2

        for i in range(n):
            # Window boundaries
            start_idx = max(0, i - half_w)
            end_idx = min(n, i + half_w + 1)
            window = lab_frames[start_idx:end_idx]

            # Gaussian temporal weights
            distances = np.arange(start_idx - i, end_idx - i)
            weights = np.exp(- (distances ** 2) / (2.0 * 1.5 ** 2))
            weights /= np.sum(weights)

            # Compute weighted temporal average
            avg_lab = np.zeros((h, w, 3), dtype=np.float32)
            for f_lab, w_t in zip(window, weights):
                avg_lab += f_lab * w_t

            # Blend luminance L channel according to filter strength
            cur_l = lab_frames[i][:, :, 0]
            avg_l = avg_lab[:, :, 0]
            final_l = (1.0 - self.frequency_filter_strength) * cur_l + self.frequency_filter_strength * avg_l

            # Chrominance (A & B) channels receive light temporal smoothing
            cur_ab = lab_frames[i][:, :, 1:]
            avg_ab = avg_lab[:, :, 1:]
            final_ab = 0.65 * cur_ab + 0.35 * avg_ab

            merged = np.dstack([final_l, final_ab])
            merged = np.clip(merged, 0, 255).astype(np.uint8)
            out_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

            # Edge clamping to restore sharp manga inking
            if self.preserve_lineart_edges:
                out_rgb = self._restore_lines(frames[i], out_rgb)

            smoothed_lab.append(out_rgb)

        return smoothed_lab

    def _restore_lines(self, original_rgb: np.ndarray, smoothed_rgb: np.ndarray) -> np.ndarray:
        """Protect pure dark lineart from temporal ghosting."""
        gray_orig = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        # Identify dark ink strokes
        line_mask = (gray_orig < 55)[:, :, np.newaxis]
        restored = np.where(line_mask, original_rgb, smoothed_rgb)
        return restored
