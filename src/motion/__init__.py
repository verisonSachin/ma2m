"""Motion package: timeline director, camera trajectory controller, and I2V motion diffusion engine."""

from src.motion.timeline_director import TimelineDirector, TimelineConfig, ShotConfig
from src.motion.camera_controller import CameraController, CameraMotionProfile
from src.motion.i2v_diffusion import I2VMotionEngine

__all__ = [
    "TimelineDirector",
    "TimelineConfig",
    "ShotConfig",
    "CameraController",
    "CameraMotionProfile",
    "I2VMotionEngine",
]
