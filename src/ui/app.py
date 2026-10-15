"""Full interactive Gradio WebUI for Manga2Motion."""

import os
import tempfile
from typing import List
import gradio as gr
import numpy as np
import cv2

from src.ingestion.panel_extractor import PanelExtractor, Panel
from src.ingestion.bubble_detector import BubbleDetector
from src.ingestion.inpainter import MangaInpainter
from src.colorization.lineart_cleaner import LineartCleaner
from src.colorization.palette_manager import PaletteManager, CharacterPalette
from src.colorization.colorizer_diffusion import MangaColorizer
from src.decomposition.sam2_segmenter import SAM2Segmenter
from src.decomposition.depth_estimator import DepthEstimator
from src.motion.timeline_director import TimelineDirector, ShotConfig
from src.motion.camera_controller import CameraMotionProfile
from src.motion.i2v_diffusion import I2VMotionEngine
from src.postprocessing.interpolator import FrameInterpolator
from src.postprocessing.deflicker import TemporalDeflicker
from src.postprocessing.video_encoder import VideoEncoder
from src.ui.components.panel_viewer import render_panel_gallery
from src.ui.components.palette_picker import render_palette_controls
from src.ui.components.timeline_view import render_timeline_controls


class Manga2MotionUI:
    """Stateful coordinator for the Manga2Motion interactive application."""

    def __init__(self):
        self.panel_extractor = PanelExtractor()
        self.bubble_detector = BubbleDetector()
        self.inpainter = MangaInpainter()
        self.lineart_cleaner = LineartCleaner()
        self.palette_manager = PaletteManager()
        self.colorizer = MangaColorizer()
        self.segmenter = SAM2Segmenter()
        self.depth_estimator = DepthEstimator()
        self.motion_engine = I2VMotionEngine()
        self.interpolator = FrameInterpolator()
        self.deflicker = TemporalDeflicker()
        self.encoder = VideoEncoder(fps=24)

        self.extracted_panels: List[Panel] = []
        self.active_colorized_panel: np.ndarray = None

    def handle_extract_panels(self, file_paths):
        """Extract panels and return annotated page + panel crops."""
        if not file_paths:
            return None, []

        page_paths = [f.name if hasattr(f, "name") else str(f) for f in file_paths]
        self.extracted_panels = self.panel_extractor.process_pages(page_paths)

        if not self.extracted_panels:
            return None, []

        # Annotate first page
        annotated_first = self.panel_extractor.annotate_panels(
            page_paths[0],
            [p for p in self.extracted_panels if p.page_idx == 0],
        )

        crops = [p.image for p in self.extracted_panels]
        return annotated_first, crops

    def handle_inpaint_preview(self, panel_idx: int = 0):
        """Inpaint bubbles from selected panel."""
        if not self.extracted_panels:
            return None
        idx = min(panel_idx, len(self.extracted_panels) - 1)
        target_panel = self.extracted_panels[idx].image

        mask, _ = self.bubble_detector.detect(target_panel)
        clean = self.inpainter.inpaint(target_panel, mask)
        return clean

    def handle_extract_palette(self, ref_image):
        """Extract K-Means dominant colors from reference artwork."""
        if ref_image is None:
            return "#C0392B", "#F5CBA7", "#2980B9", "#1B2631", "#F39C12"

        colors = self.palette_manager.extract_palette_from_image(ref_image, k=5)
        while len(colors) < 5:
            colors.append("#333333")
        return colors[0], colors[1], colors[2], colors[3], colors[4]

    def handle_colorize_preview(
        self,
        hair: str,
        skin: str,
        eyes: str,
        outfit: str,
        accent: str,
        ref_image,
    ):
        """Generate a colorized preview of the primary focal panel."""
        if not self.extracted_panels:
            demo = np.ones((512, 512, 3), dtype=np.uint8) * 240
            cv2.putText(demo, "Please upload manga pages first", (30, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
            return demo

        palette = CharacterPalette(
            hair=hair,
            skin=skin,
            eyes=eyes,
            primary_clothing=outfit,
            accent=accent,
        )

        panel_img = self.extracted_panels[0].image
        mask, _ = self.bubble_detector.detect(panel_img)
        clean_img = self.inpainter.inpaint(panel_img, mask)

        colorized = self.colorizer.colorize(
            clean_img,
            palette=palette,
            reference_image=ref_image,
        )
        self.active_colorized_panel = colorized
        return colorized

    def handle_render_video(
        self,
        timeline_mode_val: str,
        fps_val: int,
        preset_name: str,
        prompt_val: str,
        use_deflicker: bool,
        use_interp: bool,
    ):
        """Execute end-to-end 30-frame video generation."""
        if not self.extracted_panels:
            mock_img = np.ones((512, 768, 3), dtype=np.uint8) * 245
            cv2.rectangle(mock_img, (100, 100), (668, 412), (30, 30, 30), 4)
            cv2.putText(mock_img, "MANGA2MOTION SAMPLE", (180, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
            mock_panel = Panel(page_idx=0, panel_idx=0, global_order=0, bbox=(0, 0, 768, 512), image=mock_img)
            self.extracted_panels = [mock_panel]

        director = TimelineDirector(total_target_frames=30, fps=int(fps_val), mode=timeline_mode_val)
        timeline = director.build_timeline(self.extracted_panels)

        all_rendered_frames: List[np.ndarray] = []
        depth_preview = None

        profile = CameraMotionProfile()
        if "dolly_in" in preset_name:
            profile = CameraMotionProfile(zoom_start=1.0, zoom_end=1.28, pan_y=(0.0, -0.05))
        elif "dynamic_action" in preset_name:
            profile = CameraMotionProfile(zoom_start=1.05, zoom_end=1.35, pan_x=(-0.15, 0.2), rotation_degrees=1.8)
        elif "impact" in preset_name:
            profile = CameraMotionProfile(zoom_start=1.2, zoom_end=1.12, shake_intensity=0.06)
        elif "parallax_pan" in preset_name:
            profile = CameraMotionProfile(pan_x=(-0.2, 0.2))
        elif "dutch_angle" in preset_name:
            profile = CameraMotionProfile(rotation_degrees=5.0, zoom_start=1.0, zoom_end=1.15)

        for shot in timeline.shots:
            panel = self.extracted_panels[shot.source_panel_idx]
            panel_rgb = panel.image

            mask, _ = self.bubble_detector.detect(panel_rgb)
            clean_rgb = self.inpainter.inpaint(panel_rgb, mask)

            color_rgb = self.colorizer.colorize(clean_rgb)

            layers = self.segmenter.segment_layers(color_rgb)
            depth = self.depth_estimator.estimate_depth(color_rgb)
            depth_preview = self.depth_estimator.visualize_depth(depth)

            shot_frames = self.motion_engine.generate_motion(
                keyframe_rgb=color_rgb,
                depth_map=depth,
                motion_profile=profile,
                motion_prompt=prompt_val,
                frame_count=shot.frame_count,
                layers=layers,
            )
            all_rendered_frames.extend(shot_frames)

        if use_interp or len(all_rendered_frames) != 30:
            all_rendered_frames = self.interpolator.interpolate_sequence(
                all_rendered_frames,
                target_frame_count=30,
            )

        if use_deflicker:
            all_rendered_frames = self.deflicker.deflicker_sequence(all_rendered_frames)

        out_dir = tempfile.gettempdir()
        out_path = os.path.join(out_dir, "manga2motion_30frames.mp4")
        self.encoder.export_mp4(all_rendered_frames, out_path, fps=int(fps_val))

        return out_path, depth_preview


def build_gradio_app() -> gr.Blocks:
    """Build the Gradio application layout and interaction bindings."""
    ui = Manga2MotionUI()

    theme = gr.themes.Soft(
        primary_hue="rose",
        secondary_hue="slate",
        neutral_hue="zinc",
    )

    with gr.Blocks(title="Manga2Motion - AI Manga-to-Video Diffusion System", theme=theme) as demo:
        gr.Markdown(
            """
            # 🎬 Manga2Motion: AI-Powered Manga-to-Video Diffusion
            **Convert multi-page static manga into temporally cohesive 30-frame anime motion video.**
            """
        )

        with gr.Tabs():
            with gr.TabItem("Step 1: Manga Ingestion & Reading Order"):
                p_comps = render_panel_gallery()

            with gr.TabItem("Step 2: Reference Colorization"):
                c_comps = render_palette_controls()

            with gr.TabItem("Step 3: Timeline & 30-Frame Video Generation"):
                t_comps = render_timeline_controls()

        p_comps["btn_extract"].click(
            fn=ui.handle_extract_panels,
            inputs=[p_comps["page_uploads"]],
            outputs=[p_comps["annotated_page_view"], p_comps["panel_gallery"]],
        )

        p_comps["inpaint_toggle"].change(
            fn=ui.handle_inpaint_preview,
            inputs=[],
            outputs=[p_comps["inpainted_preview"]],
        )

        c_comps["btn_extract_palette"].click(
            fn=ui.handle_extract_palette,
            inputs=[c_comps["ref_image_upload"]],
            outputs=[
                c_comps["hair_color"],
                c_comps["skin_color"],
                c_comps["eyes_color"],
                c_comps["outfit_color"],
                c_comps["accent_color"],
            ],
        )

        c_comps["btn_preview_color"].click(
            fn=ui.handle_colorize_preview,
            inputs=[
                c_comps["hair_color"],
                c_comps["skin_color"],
                c_comps["eyes_color"],
                c_comps["outfit_color"],
                c_comps["accent_color"],
                c_comps["ref_image_upload"],
            ],
            outputs=[c_comps["colorized_preview"]],
        )

        t_comps["btn_render_video"].click(
            fn=ui.handle_render_video,
            inputs=[
                t_comps["timeline_mode"],
                t_comps["fps_slider"],
                t_comps["motion_preset"],
                t_comps["motion_prompt"],
                t_comps["deflicker_toggle"],
                t_comps["rife_interp_toggle"],
            ],
            outputs=[t_comps["video_player"], t_comps["depth_player"]],
        )

    return demo


def main():
    """CLI entrypoint for launching Gradio UI."""
    app = build_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
