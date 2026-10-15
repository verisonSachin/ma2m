"""Temporal frame interpolation (RIFE / FILM / Dense Optical Flow) to reach exactly 30 frames."""

import os
from typing import List, Optional, Union
import numpy as np
import cv2
import torch


class FrameInterpolator:
    """Interpolates motion sequences up to target frame count with edge-aware optical flow."""

    def __init__(
        self,
        engine: str = "rife",
        model_path: Optional[str] = "models/rife46.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.engine = engine.lower()
        self.model_path = model_path
        self.device = device
        self.rife_model = None

        if self.engine == "rife" and self.model_path and os.path.exists(self.model_path):
            self._init_rife()

    def _init_rife(self):
        """Attempts to load RIFE neural flow model."""
        try:
            print(f"[FrameInterpolator] Loading RIFE weights from {self.model_path}...")
            self.rife_model = torch.jit.load(self.model_path, map_location=self.device)
            self.rife_model.eval()
            print("[FrameInterpolator] RIFE model successfully loaded.")
        except Exception as e:
            print(f"[FrameInterpolator] Could not load RIFE model ({e}). Using DIS optical flow interpolator.")
            self.rife_model = None

    def interpolate_sequence(
        self,
        frames: List[np.ndarray],
        target_frame_count: int = 30,
    ) -> List[np.ndarray]:
        """
        Interpolates arbitrary length frame sequence up to exactly target_frame_count.

        Args:
            frames: Input list of RGB uint8 images.
            target_frame_count: Target number of frames (30).

        Returns:
            interpolated: List of exactly target_frame_count frames.
        """
        n_in = len(frames)
        if n_in == target_frame_count:
            return frames
        if n_in < 2:
            return [frames[0].copy() for _ in range(target_frame_count)]

        # Time parameter for target frames normalized [0.0, n_in - 1]
        target_timestamps = np.linspace(0, n_in - 1, target_frame_count)
        output_frames = []

        for t in target_timestamps:
            idx0 = int(np.floor(t))
            idx1 = min(idx0 + 1, n_in - 1)
            alpha = float(t - idx0)

            if idx0 == idx1 or alpha < 1e-4:
                output_frames.append(frames[idx0].copy())
            elif alpha > (1.0 - 1e-4):
                output_frames.append(frames[idx1].copy())
            else:
                f0 = frames[idx0]
                f1 = frames[idx1]

                if self.rife_model is not None:
                    inter_frame = self._interpolate_rife(f0, f1, alpha)
                else:
                    inter_frame = self._interpolate_flow(f0, f1, alpha)

                output_frames.append(inter_frame)

        return output_frames

    def _interpolate_rife(self, img0: np.ndarray, img1: np.ndarray, alpha: float) -> np.ndarray:
        """Inference step with RIFE neural network."""
        t0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        timestep = torch.tensor([alpha], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            res = self.rife_model(t0, t1, timestep)
            res = torch.clamp(res, 0.0, 1.0)
            np_out = (res[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        return np_out

    def _interpolate_flow(self, img0: np.ndarray, img1: np.ndarray, alpha: float) -> np.ndarray:
        """
        Bidirectional dense optical flow interpolation using OpenCV DIS flow.
        Warps img0 forward and img1 backward along flow vectors and blends.
        """
        h, w = img0.shape[:2]
        gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        # Compute forward flow: img0 -> img1
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow_fwd = dis.calc(gray0, gray1, None)

        # Compute backward flow: img1 -> img0
        flow_bwd = dis.calc(gray1, gray0, None)

        # Coordinate grid
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Forward warp: img0 pushed forward by alpha * flow_fwd
        map0_x = grid_x + alpha * flow_fwd[:, :, 0]
        map0_y = grid_y + alpha * flow_fwd[:, :, 1]
        warped0 = cv2.remap(img0, map0_x, map0_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Backward warp: img1 pulled backward by (1 - alpha) * flow_bwd
        map1_x = grid_x + (1.0 - alpha) * flow_bwd[:, :, 0]
        map1_y = grid_y + (1.0 - alpha) * flow_bwd[:, :, 1]
        warped1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Non-linear alpha blend between the two warps
        blended = ((1.0 - alpha) * warped0.astype(np.float32) + alpha * warped1.astype(np.float32)).astype(np.uint8)
        return blended
