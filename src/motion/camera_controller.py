"""2.5D parallax camera trajectory controller and depth-guided perspective warping."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import numpy as np
import cv2

from src.decomposition.sam2_segmenter import LayerDecomposition


@dataclass
class CameraMotionProfile:
    """Defines camera transformation parameters over a sequence."""
    zoom_start: float = 1.0
    zoom_end: float = 1.2
    pan_x: Tuple[float, float] = (0.0, 0.0)  # (start_fraction, end_fraction)
    pan_y: Tuple[float, float] = (0.0, 0.0)
    rotation_degrees: float = 0.0
    shake_intensity: float = 0.0
    shake_decay: float = 0.85
    easing: str = "cubic_ease_in_out"


class CameraController:
    """Calculates continuous camera curves and renders depth-aware 2.5D parallax frames."""

    def __init__(self):
        self._easing_map = {
            "linear": self._ease_linear,
            "cubic_ease_in_out": self._ease_cubic_in_out,
            "exponential_in": self._ease_exp_in,
            "elastic_out": self._ease_elastic_out,
            "sine_ease_in_out": self._ease_sine_in_out,
        }

    def generate_shot_frames(
        self,
        image_rgb: np.ndarray,
        depth_map: np.ndarray,
        motion_profile: CameraMotionProfile,
        num_frames: int,
        layers: Optional[LayerDecomposition] = None,
    ) -> List[np.ndarray]:
        """
        Synthesize motion frames using 2.5D depth-guided parallax and camera curves.

        Args:
            image_rgb: Base colorized panel (H, W, 3).
            depth_map: Normalized depth map (H, W) [0.0 = near, 1.0 = far].
            motion_profile: Camera trajectory parameters.
            num_frames: Total number of frames to generate for this shot.
            layers: Optional decomposed foreground/background layers for occluded multi-plane parallax.

        Returns:
            frames: List of RGB uint8 images of length num_frames.
        """
        h, w = image_rgb.shape[:2]
        ease_func = self._easing_map.get(motion_profile.easing, self._ease_cubic_in_out)

        frames = []
        for f in range(num_frames):
            t = f / float(max(1, num_frames - 1))
            progress = ease_func(t)

            # Interpolate zoom
            current_zoom = motion_profile.zoom_start + progress * (motion_profile.zoom_end - motion_profile.zoom_start)

            # Interpolate pans (in pixels)
            px_start = motion_profile.pan_x[0] * w
            px_end = motion_profile.pan_x[1] * w
            current_pan_x = px_start + progress * (px_end - px_start)

            py_start = motion_profile.pan_y[0] * h
            py_end = motion_profile.pan_y[1] * h
            current_pan_y = py_start + progress * (py_end - py_start)

            # Interpolate rotation
            current_rot = motion_profile.rotation_degrees * progress

            # Harmonic decaying shake
            if motion_profile.shake_intensity > 0:
                decay = motion_profile.shake_decay ** f
                shake_x = np.sin(f * 2.8) * motion_profile.shake_intensity * w * decay
                shake_y = np.cos(f * 3.4) * motion_profile.shake_intensity * h * decay
                current_pan_x += shake_x
                current_pan_y += shake_y

            # Render frame with depth parallax
            if layers is not None:
                frame = self._render_multiplane_layer(
                    layers,
                    zoom=current_zoom,
                    pan_x=current_pan_x,
                    pan_y=current_pan_y,
                    rotation=current_rot,
                )
            else:
                frame = self._render_depth_warp(
                    image_rgb,
                    depth_map,
                    zoom=current_zoom,
                    pan_x=current_pan_x,
                    pan_y=current_pan_y,
                    rotation=current_rot,
                )

            frames.append(frame)

        return frames

    def _render_depth_warp(
        self,
        image_rgb: np.ndarray,
        depth_map: np.ndarray,
        zoom: float,
        pan_x: float,
        pan_y: float,
        rotation: float,
    ) -> np.ndarray:
        """Warps image pixels with depth-dependent parallax offset."""
        h, w = image_rgb.shape[:2]

        # Near objects (depth ~ 0.0) move more with camera pan, far background (depth ~ 1.0) moves less
        # Parallax factor: 1.0 (foreground) down to 0.35 (background)
        parallax_factor = 1.0 - 0.65 * depth_map

        # Base affine transform for camera center
        center_x, center_y = w / 2.0, h / 2.0
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rotation, zoom)

        # Apply global affine warp
        warped_base = cv2.warpAffine(
            image_rgb,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Depth-dependent pixel displacement grid
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Shift coordinates based on depth
        map_x = grid_x + (pan_x * parallax_factor).astype(np.float32)
        map_y = grid_y + (pan_y * parallax_factor).astype(np.float32)

        # Remap image
        final_frame = cv2.remap(
            warped_base,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return final_frame

    def _render_multiplane_layer(
        self,
        layers: LayerDecomposition,
        zoom: float,
        pan_x: float,
        pan_y: float,
        rotation: float,
    ) -> np.ndarray:
        """Renders 2.5D multi-plane scene: Foreground character plate moves faster than background plate."""
        h, w = layers.background_rgb.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 1. Background plate moves slowly (0.4x pan, subtle zoom)
        bg_zoom = 1.0 + (zoom - 1.0) * 0.4
        bg_rot_mat = cv2.getRotationMatrix2D(center, rotation * 0.3, bg_zoom)
        bg_rot_mat[0, 2] += pan_x * 0.3
        bg_rot_mat[1, 2] += pan_y * 0.3

        bg_rendered = cv2.warpAffine(
            layers.background_rgb,
            bg_rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # 2. Foreground plate moves dynamically (1.0x pan, full zoom)
        fg_rot_mat = cv2.getRotationMatrix2D(center, rotation, zoom)
        fg_rot_mat[0, 2] += pan_x
        fg_rot_mat[1, 2] += pan_y

        fg_rendered = cv2.warpAffine(
            layers.foreground_rgba,
            fg_rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # Alpha composite foreground over background
        fg_rgb = fg_rendered[:, :, :3]
        fg_alpha = (fg_rendered[:, :, 3] / 255.0)[:, :, np.newaxis]

        composite = (fg_alpha * fg_rgb.astype(np.float32) + (1.0 - fg_alpha) * bg_rendered.astype(np.float32)).astype(np.uint8)
        return composite

    # Easing equations
    @staticmethod
    def _ease_linear(t: float) -> float:
        return t

    @staticmethod
    def _ease_cubic_in_out(t: float) -> float:
        if t < 0.5:
            return 4.0 * t * t * t
        p = 2.0 * t - 2.0
        return 0.5 * p * p * p + 1.0

    @staticmethod
    def _ease_exp_in(t: float) -> float:
        return 0.0 if t == 0.0 else float(2.0 ** (10.0 * (t - 1.0)))

    @staticmethod
    def _ease_elastic_out(t: float) -> float:
        if t == 0.0:
            return 0.0
        if t == 1.0:
            return 1.0
        p = 0.3
        return float(2.0 ** (-10.0 * t) * np.sin((t - p / 4.0) * (2.0 * np.pi) / p) + 1.0)

    @staticmethod
    def _ease_sine_in_out(t: float) -> float:
        return float(-0.5 * (np.cos(np.pi * t) - 1.0))
