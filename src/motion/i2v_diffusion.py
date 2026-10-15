"""Image-to-Video (I2V) Motion Diffusion Engine (AnimateDiff / SVD / Neural Motion)."""

import os
from typing import List, Optional, Union, Tuple
import numpy as np
import cv2
import torch
from PIL import Image

from src.motion.camera_controller import CameraController, CameraMotionProfile
from src.decomposition.sam2_segmenter import LayerDecomposition


class I2VMotionEngine:
    """Generates continuous motion video sequences from single colorized manga keyframes."""

    def __init__(
        self,
        engine: str = "animatediff",
        motion_adapter_model: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        base_model: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_native_frames: int = 16,
    ):
        self.engine = engine
        self.motion_adapter_model = motion_adapter_model
        self.base_model = base_model
        self.device = device
        self.num_native_frames = num_native_frames

        self.pipeline = None
        self.camera_controller = CameraController()

        self._init_pipeline()

    def _init_pipeline(self):
        """Initializes AnimateDiff motion diffusion pipeline if GPU and packages are ready."""
        if not torch.cuda.is_available():
            print("[I2VMotionEngine] GPU unavailable. Using camera parallax motion engine.")
            return

        try:
            from diffusers import (
                AnimateDiffPipeline,
                MotionAdapter,
                DDIMScheduler,
            )

            print(f"[I2VMotionEngine] Loading MotionAdapter: {self.motion_adapter_model}...")
            adapter = MotionAdapter.from_pretrained(
                self.motion_adapter_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            print(f"[I2VMotionEngine] Loading AnimateDiff pipeline: {self.base_model}...")
            self.pipeline = AnimateDiffPipeline.from_pretrained(
                self.base_model,
                motion_adapter=adapter,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config,
                clip_sample=False,
                timestep_spacing="linspace",
                steps_offset=1,
            )
            self.pipeline.to(self.device)
            self.pipeline.enable_vae_slicing()
            print("[I2VMotionEngine] AnimateDiff pipeline initialized.")
        except Exception as e:
            print(f"[I2VMotionEngine] Could not initialize AnimateDiff ({e}). Falling back to camera trajectory parallax.")
            self.pipeline = None

    def generate_motion(
        self,
        keyframe_rgb: np.ndarray,
        depth_map: np.ndarray,
        motion_profile: CameraMotionProfile,
        motion_prompt: str,
        frame_count: int = 16,
        layers: Optional[LayerDecomposition] = None,
    ) -> List[np.ndarray]:
        """
        Generate continuous motion frames.

        Args:
            keyframe_rgb: The starting colorized keyframe (H, W, 3).
            depth_map: Corresponding depth map.
            motion_profile: Trajectory profile (pan, zoom, tilt, shake).
            motion_prompt: Diffusion text prompt describing the action.
            frame_count: Target raw frame count before final interpolation.
            layers: Optional decomposed visual layers.

        Returns:
            frames: List of RGB uint8 frames.
        """
        if self.pipeline is not None and frame_count == self.num_native_frames:
            try:
                return self._infer_animatediff(keyframe_rgb, motion_prompt, frame_count)
            except Exception as e:
                print(f"[I2VMotionEngine] Diffusion motion inference failed ({e}). Using camera trajectory.")

        # Camera trajectory & parallax motion synthesis
        frames = self.camera_controller.generate_shot_frames(
            image_rgb=keyframe_rgb,
            depth_map=depth_map,
            motion_profile=motion_profile,
            num_frames=frame_count,
            layers=layers,
        )

        # If it's a high-action motion, apply kinetic anime speed streaks overlay
        if "action" in motion_prompt.lower() or "slashing" in motion_prompt.lower():
            frames = self._apply_action_speedlines(frames)

        return frames

    def _infer_animatediff(
        self,
        keyframe_rgb: np.ndarray,
        prompt: str,
        frame_count: int,
    ) -> List[np.ndarray]:
        """Run native AnimateDiff I2V generation."""
        h, w = keyframe_rgb.shape[:2]
        # Ensure 512x512 multiples for standard SD1.5 motion adapters
        gen_w = (w // 64) * 64
        gen_h = (h // 64) * 64

        full_prompt = f"masterpiece, anime aesthetic, {prompt}, 4k, smooth animation"
        neg_prompt = "static, photo, low quality, jitter, stutter, artifacts"

        generator = torch.Generator(device=self.device).manual_seed(42)
        output = self.pipeline(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            num_frames=frame_count,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=generator,
            height=gen_h,
            width=gen_w,
        )

        frames = []
        for pil_frame in output.frames[0]:
            frame_np = np.array(pil_frame.convert("RGB"))
            if (gen_h, gen_w) != (h, w):
                frame_np = cv2.resize(frame_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame_np)

        return frames

    def _apply_action_speedlines(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Add dynamic anime action lines and subtle directional blur to climax frames."""
        h, w = frames[0].shape[:2]
        enhanced = []

        for idx, f in enumerate(frames):
            frame_copy = f.copy().astype(np.float32)
            # Speedlines intensity increases midway through the shot
            intensity = np.sin((idx / float(len(frames))) * np.pi) * 0.35

            if intensity > 0.05:
                # Generate procedural radial/horizontal speed streaks
                noise = np.random.uniform(0.0, 1.0, (h, 1)).astype(np.float32)
                streaks = np.repeat(noise, w, axis=1)
                streaks = (streaks > 0.88).astype(np.float32) * 255.0
                streaks = cv2.GaussianBlur(streaks, (15, 1), 0)

                streak_3c = np.repeat(streaks[:, :, np.newaxis], 3, axis=2)
                blended = np.clip(frame_copy + streak_3c * intensity, 0, 255).astype(np.uint8)
                enhanced.append(blended)
            else:
                enhanced.append(f)

        return enhanced
