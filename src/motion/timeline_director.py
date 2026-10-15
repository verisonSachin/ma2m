"""Timeline scene director: 30-frame budget allocation and shot sequence planning."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from src.ingestion.panel_extractor import Panel


@dataclass
class ShotConfig:
    """Configuration for a single shot within the 30-frame timeline budget."""
    shot_id: int
    source_panel_idx: int
    frame_range: Tuple[int, int]  # (start_frame, end_frame) inclusive
    frame_count: int
    motion_type: str  # "camera_dolly_in", "dynamic_action_dash", "impact_freeze_settle", "parallax_pan_right"
    camera_speed: float = 1.0
    motion_prompt: str = ""
    transition_to_next: str = "cut"  # "cut", "crossfade", "whip_pan"
    metadata: Dict = field(default_factory=dict)


@dataclass
class TimelineConfig:
    """Master timeline specification matching timeline.json schema."""
    scene_id: str = "manga_scene_001"
    source_pages: List[str] = field(default_factory=list)
    total_target_frames: int = 30
    fps: int = 24
    color_profile: Dict = field(default_factory=dict)
    shots: List[ShotConfig] = field(default_factory=list)

    def to_dict(self) -> Dict:
        data = asdict(self)
        # Ensure tuples are converted to lists for clean JSON serialization
        for shot in data["shots"]:
            shot["frame_range"] = list(shot["frame_range"])
        return data

    def save_json(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "TimelineConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        shots = []
        for s in data.get("shots", []):
            shots.append(
                ShotConfig(
                    shot_id=s["shot_id"],
                    source_panel_idx=s["source_panel_idx"],
                    frame_range=tuple(s["frame_range"]),
                    frame_count=s["frame_count"],
                    motion_type=s.get("motion_type", "camera_dolly_in"),
                    camera_speed=s.get("camera_speed", 1.0),
                    motion_prompt=s.get("motion_prompt", ""),
                    transition_to_next=s.get("transition_to_next", "cut"),
                    metadata=s.get("metadata", {}),
                )
            )

        return cls(
            scene_id=data.get("scene_id", "manga_scene_001"),
            source_pages=data.get("source_pages", []),
            total_target_frames=data.get("total_target_frames", 30),
            fps=data.get("fps", 24),
            color_profile=data.get("color_profile", {}),
            shots=shots,
        )


class TimelineDirector:
    """Calculates and budgets frames across extracted panels to strictly satisfy the 30-frame limit."""

    def __init__(
        self,
        total_target_frames: int = 30,
        fps: int = 24,
        mode: str = "climax_continuity",  # "climax_continuity" or "multi_panel_montage"
    ):
        self.total_target_frames = total_target_frames
        self.fps = fps
        self.mode = mode

    def build_timeline(
        self,
        panels: List[Panel],
        source_page_names: Optional[List[str]] = None,
        focal_panel_idx: Optional[int] = None,
        color_profile_dict: Optional[Dict] = None,
    ) -> TimelineConfig:
        """
        Build a deterministic 30-frame timeline budget according to chosen mode.
        """
        if not panels:
            raise ValueError("No manga panels provided to timeline director.")

        source_pages = source_page_names or [f"page_{p.page_idx + 1:02d}.png" for p in panels]
        # Remove duplicate page names while preserving order
        unique_pages = list(dict.fromkeys(source_pages))

        if self.mode == "climax_continuity":
            shots = self._build_mode_a_climax(panels, focal_panel_idx)
        else:
            shots = self._build_mode_b_montage(panels)

        # Enforce exact frame budget verification
        allocated_frames = sum(s.frame_count for s in shots)
        if allocated_frames != self.total_target_frames:
            # Rebalance last shot to match exactly
            diff = self.total_target_frames - allocated_frames
            shots[-1].frame_count += diff
            shots[-1].frame_range = (
                shots[-1].frame_range[0],
                shots[-1].frame_range[0] + shots[-1].frame_count - 1,
            )

        return TimelineConfig(
            scene_id="manga_scene_001",
            source_pages=unique_pages,
            total_target_frames=self.total_target_frames,
            fps=self.fps,
            color_profile=color_profile_dict or {},
            shots=shots,
        )

    def _build_mode_a_climax(self, panels: List[Panel], focal_panel_idx: Optional[int]) -> List[ShotConfig]:
        """
        Mode A: Climax Action Continuity.
        A single high-impact panel receives the full 30-frame animation budget.
        """
        # If no focal panel specified, select the panel with the largest bounding box area (typically the climax splash)
        if focal_panel_idx is None or focal_panel_idx >= len(panels):
            focal_panel_idx = max(range(len(panels)), key=lambda i: panels[i].width * panels[i].height)

        shot = ShotConfig(
            shot_id=1,
            source_panel_idx=focal_panel_idx,
            frame_range=(0, self.total_target_frames - 1),
            frame_count=self.total_target_frames,
            motion_type="dynamic_action_dash",
            camera_speed=1.5,
            motion_prompt="continuous fluid action sequence, dramatic forward thrust, dynamic anime motion blur, cinematic lighting",
            transition_to_next="none",
        )
        return [shot]

    def _build_mode_b_montage(self, panels: List[Panel]) -> List[ShotConfig]:
        """
        Mode B: Multi-Panel Scene Montage (Narrative Arc).
        Allocates frames across 3 key narrative beats: Setup (8f) -> Climax (14f) -> Settle (8f).
        """
        n = len(panels)
        if n == 1:
            return self._build_mode_a_climax(panels, 0)
        elif n == 2:
            # 2 panels: 14 frames Setup, 16 frames Action
            shot1 = ShotConfig(
                shot_id=1,
                source_panel_idx=0,
                frame_range=(0, 13),
                frame_count=14,
                motion_type="camera_dolly_in",
                camera_speed=1.0,
                motion_prompt="slow dramatic camera push-in, intense eye focus, wind fluttering",
                transition_to_next="crossfade",
            )
            shot2 = ShotConfig(
                shot_id=2,
                source_panel_idx=1,
                frame_range=(14, 29),
                frame_count=16,
                motion_type="dynamic_action_dash",
                camera_speed=2.0,
                motion_prompt="dynamic combat motion, explosive strike, anime action lines",
                transition_to_next="none",
            )
            return [shot1, shot2]

        # 3 or more panels: standard 3-shot narrative arc
        # Pick setup panel (early panel), climax panel (largest/central), reaction panel (late panel)
        setup_idx = 0
        climax_idx = max(range(1, n - 1 if n > 2 else n), key=lambda i: panels[i].width * panels[i].height)
        reaction_idx = n - 1

        shots = [
            ShotConfig(
                shot_id=1,
                source_panel_idx=setup_idx,
                frame_range=(0, 7),
                frame_count=8,
                motion_type="camera_dolly_in",
                camera_speed=1.1,
                motion_prompt="slow dramatic camera push-in, hair fluttering in wind, cinematic anime lighting",
                transition_to_next="cut",
            ),
            ShotConfig(
                shot_id=2,
                source_panel_idx=climax_idx,
                frame_range=(8, 21),
                frame_count=14,
                motion_type="dynamic_action_dash",
                camera_speed=2.2,
                motion_prompt="rapid forward motion, slashing sword attack, motion blur, glowing blade trail",
                transition_to_next="cut",
            ),
            ShotConfig(
                shot_id=3,
                source_panel_idx=reaction_idx,
                frame_range=(22, 29),
                frame_count=8,
                motion_type="impact_freeze_settle",
                camera_speed=0.9,
                motion_prompt="shockwave expanding, dust particles settling, intense character gaze",
                transition_to_next="none",
            ),
        ]
        return shots
