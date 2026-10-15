"""CLI Entrypoint for Manga2Motion: Automated Manga-to-Video Diffusion Pipeline."""

import argparse
import os
import sys
import yaml
from typing import List
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.panel_extractor import PanelExtractor, Panel
from src.ingestion.bubble_detector import BubbleDetector
from src.ingestion.inpainter import MangaInpainter
from src.colorization.palette_manager import PaletteManager, CharacterPalette
from src.colorization.colorizer_diffusion import MangaColorizer
from src.decomposition.sam2_segmenter import SAM2Segmenter
from src.decomposition.depth_estimator import DepthEstimator
from src.motion.timeline_director import TimelineDirector, TimelineConfig
from src.motion.camera_controller import CameraMotionProfile
from src.motion.i2v_diffusion import I2VMotionEngine
from src.postprocessing.interpolator import FrameInterpolator
from src.postprocessing.deflicker import TemporalDeflicker
from src.postprocessing.video_encoder import VideoEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manga2Motion CLI: End-to-end Manga-to-Video Diffusion System.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        required=True,
        help="Paths to ~3 consecutive manga page images (PNG/JPG).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Strict total frame budget for output video sequence.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target video framerate (e.g. 24 or 30).",
    )
    parser.add_argument(
        "--mode",
        choices=["climax_continuity", "multi_panel_montage"],
        default="climax_continuity",
        help="Timeline synthesis mode (Mode A: climax action, Mode B: multi-panel montage).",
    )
    parser.add_argument(
        "--motion-preset",
        choices=["camera_dolly_in", "dynamic_action_dash", "impact_freeze_settle", "parallax_pan_right", "dutch_angle_tilt"],
        default="dynamic_action_dash",
        help="Camera trajectory preset.",
    )
    parser.add_argument(
        "--motion-prompt",
        type=str,
        default="slashing sword strike, rapid anime action, dynamic camera, glowing particle trails",
        help="Text prompt conditioning for motion diffusion.",
    )
    parser.add_argument(
        "--ref-image",
        type=str,
        default=None,
        help="Optional path to a colored character reference image.",
    )
    parser.add_argument("--hair-color", type=str, default="#C0392B", help="HEX hair color.")
    parser.add_argument("--skin-color", type=str, default="#F5CBA7", help="HEX skin color.")
    parser.add_argument("--eyes-color", type=str, default="#2980B9", help="HEX eye color.")
    parser.add_argument("--outfit-color", type=str, default="#1B2631", help="HEX outfit color.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/manga_motion_30f.mp4",
        help="Output path for final MP4 video.",
    )
    parser.add_argument(
        "--save-timeline",
        type=str,
        default="outputs/timeline.json",
        help="Path to dump timeline.json specification.",
    )
    parser.add_argument(
        "--no-inpaint",
        action="store_true",
        help="Skip speech bubble and SFX inpainting.",
    )
    parser.add_argument(
        "--no-deflicker",
        action="store_true",
        help="Skip temporal deflickering filter.",
    )
    parser.add_argument(
        "--export-prores",
        action="store_true",
        help="Also export an Apple ProRes 422 HQ master file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_pipeline.yaml",
        help="Path to pipeline configuration YAML.",
    )
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    print("==================================================================")
    print("           🎬 Manga2Motion End-to-End Pipeline CLI                ")
    print("==================================================================")
    print(f"[*] Input Pages: {len(args.pages)} scans")
    print(f"[*] Total Target Frame Budget: {args.frames} frames @ {args.fps} FPS")
    print(f"[*] Timeline Mode: {args.mode}")
    print(f"[*] Output Destination: {args.output}")

    # Phase 1: Ingestion & RTL Sequencing
    print("\n[Phase 1/6] Ingesting manga pages & extracting panels in Japanese RTL reading order...")
    panel_extractor = PanelExtractor()
    bubble_detector = BubbleDetector()
    inpainter = MangaInpainter()

    panels: List[Panel] = panel_extractor.process_pages(args.pages)
    print(f"  -> Extracted {len(panels)} panels across {len(args.pages)} pages in RTL order.")

    # Inpaint speech bubbles
    processed_panels = []
    for p in panels:
        if not args.no_inpaint:
            mask, bubbles = bubble_detector.detect(p.image)
            clean_crop = inpainter.inpaint(p.image, mask)
            p.image = clean_crop
        processed_panels.append(p)

    # Phase 2: Color Palette Locking & Colorization
    print("\n[Phase 2/6] Harmonizing character color references...")
    palette_mgr = PaletteManager()
    char_palette = CharacterPalette(
        hair=args.hair_color,
        skin=args.skin_color,
        eyes=args.eyes_color,
        primary_clothing=args.outfit_color,
        reference_image_path=args.ref_image,
    )

    if args.ref_image and os.path.exists(args.ref_image):
        print(f"  -> Extracting dominant color anchors from reference: {args.ref_image}")
        extracted_hex = palette_mgr.extract_palette_from_image(args.ref_image, k=4)
        if len(extracted_hex) >= 3:
            char_palette.hair = extracted_hex[0]
            char_palette.skin = extracted_hex[1]
            char_palette.primary_clothing = extracted_hex[2]

    colorizer = MangaColorizer()

    # Phase 3: Timeline Budgeting
    print("\n[Phase 3/6] Directing timeline shots (allocating 30 frames)...")
    director = TimelineDirector(
        total_target_frames=args.frames,
        fps=args.fps,
        mode=args.mode,
    )
    timeline: TimelineConfig = director.build_timeline(
        panels=processed_panels,
        source_page_names=args.pages,
        color_profile_dict={
            "reference_image": args.ref_image,
            "primary_character_palette": {
                "hair": char_palette.hair,
                "skin": char_palette.skin,
                "eyes": char_palette.eyes,
                "clothing": char_palette.primary_clothing,
            },
        },
    )

    if args.save_timeline:
        timeline.save_json(args.save_timeline)
        print(f"  -> Exported timeline specification to: {args.save_timeline}")

    # Phase 4: Decomposition & Depth Estimation
    print("\n[Phase 4/6] Performing layer decomposition & monocular depth estimation...")
    segmenter = SAM2Segmenter()
    depth_estimator = DepthEstimator()

    # Phase 5: I2V Motion Diffusion & Camera Controller
    print("\n[Phase 5/6] Generating camera & character motion diffusion...")
    motion_engine = I2VMotionEngine()

    # Configure camera motion profile from selected preset
    profile = CameraMotionProfile()
    if args.motion_preset == "camera_dolly_in":
        profile = CameraMotionProfile(zoom_start=1.0, zoom_end=1.25, pan_y=(0.0, -0.05), easing="cubic_ease_in_out")
    elif args.motion_preset == "dynamic_action_dash":
        profile = CameraMotionProfile(zoom_start=1.05, zoom_end=1.35, pan_x=(-0.15, 0.2), rotation_degrees=1.5, easing="exponential_in")
    elif args.motion_preset == "impact_freeze_settle":
        profile = CameraMotionProfile(zoom_start=1.2, zoom_end=1.15, shake_intensity=0.08, easing="elastic_out")
    elif args.motion_preset == "parallax_pan_right":
        profile = CameraMotionProfile(pan_x=(-0.2, 0.2), easing="linear")
    elif args.motion_preset == "dutch_angle_tilt":
        profile = CameraMotionProfile(zoom_start=1.0, zoom_end=1.12, rotation_degrees=4.5, easing="sine_ease_in_out")

    all_raw_frames: List[np.ndarray] = []

    for shot in timeline.shots:
        print(f"  -> Synthesizing Shot #{shot.shot_id} (Panel {shot.source_panel_idx + 1}): {shot.frame_count} frames [{shot.motion_type}]")
        panel = processed_panels[shot.source_panel_idx]

        # Colorize keyframe
        colorized = colorizer.colorize(panel.image, palette=char_palette, reference_image=args.ref_image)

        # Decompose & compute depth
        layers = segmenter.segment_layers(colorized)
        depth = depth_estimator.estimate_depth(colorized)

        # Generate shot frames
        shot_frames = motion_engine.generate_motion(
            keyframe_rgb=colorized,
            depth_map=depth,
            motion_profile=profile,
            motion_prompt=shot.motion_prompt or args.motion_prompt,
            frame_count=shot.frame_count,
            layers=layers,
        )
        all_raw_frames.extend(shot_frames)

    # Phase 6: Interpolation, Deflicker & Video Export
    print("\n[Phase 6/6] Polishing video sequence: interpolation, deflicker, & H.264 encode...")
    interpolator = FrameInterpolator()
    deflicker = TemporalDeflicker()
    encoder = VideoEncoder(fps=args.fps)

    # Interpolate to strict target frame budget
    print(f"  -> Interpolating sequence to exactly {args.frames} frames via optical flow...")
    final_frames = interpolator.interpolate_sequence(all_raw_frames, target_frame_count=args.frames)

    # Temporal deflickering
    if not args.no_deflicker:
        print("  -> Applying frequency-space temporal deflickering filter...")
        final_frames = deflicker.deflicker_sequence(final_frames)

    # Export MP4
    mp4_path = encoder.export_mp4(final_frames, args.output, fps=args.fps)
    print(f"\n[SUCCESS] Rendered {len(final_frames)} synchronized frames to MP4:\n  --> {mp4_path}")

    # Optional ProRes export
    if args.export_prores:
        prores_path = os.path.splitext(args.output)[0] + ".mov"
        encoder.export_prores(final_frames, prores_path, fps=args.fps)
        print(f"[SUCCESS] Exported Apple ProRes 422 HQ master:\n  --> {prores_path}")

    print("\n[DONE] Manga2Motion processing completed successfully!")


if __name__ == "__main__":
    main()
