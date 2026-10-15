"""Video encoder for high-fidelity MP4, ProRes, and WebM export."""

import os
from typing import List, Optional
import numpy as np
import cv2
import imageio


class VideoEncoder:
    """Encodes frame sequences into standard streaming MP4 (H.264) or production ProRes files."""

    def __init__(
        self,
        fps: int = 24,
        codec: str = "libx264",
        pix_fmt: str = "yuv420p",
        crf: int = 18,
    ):
        self.fps = fps
        self.codec = codec
        self.pix_fmt = pix_fmt
        self.crf = crf

    def export_mp4(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: Optional[int] = None,
    ) -> str:
        """
        Export frames as H.264 MP4.

        Args:
            frames: List of RGB uint8 arrays.
            output_path: Destination file path.
            fps: Optional framerate override.

        Returns:
            output_path: Absolute path to the created video.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        target_fps = fps or self.fps

        try:
            writer = imageio.get_writer(
                output_path,
                fps=target_fps,
                codec=self.codec,
                pixelformat=self.pix_fmt,
                ffmpeg_params=["-crf", str(self.crf), "-preset", "slow"],
            )
            for f in frames:
                writer.append_data(f)
            writer.close()
        except Exception as e:
            print(f"[VideoEncoder] imageio-ffmpeg failed ({e}). Falling back to cv2.VideoWriter.")
            self._export_cv2(frames, output_path, target_fps)

        return os.path.abspath(output_path)

    def export_prores(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: Optional[int] = None,
    ) -> str:
        """Export frames as Apple ProRes 422 HQ (ideal for video editing software)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        target_fps = fps or self.fps

        writer = imageio.get_writer(
            output_path,
            fps=target_fps,
            codec="prores",
            ffmpeg_params=["-profile:v", "3", "-pix_fmt", "yuv422p10le"],
        )
        for f in frames:
            writer.append_data(f)
        writer.close()

        return os.path.abspath(output_path)

    def export_gif(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: Optional[int] = None,
    ) -> str:
        """Export frames as an animated GIF for quick previewing."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        target_fps = fps or self.fps
        duration = 1.0 / target_fps
        imageio.mimsave(output_path, frames, duration=duration, loop=0)
        return os.path.abspath(output_path)

    def _export_cv2(self, frames: List[np.ndarray], output_path: str, fps: int):
        """Standard OpenCV VideoWriter fallback."""
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()
