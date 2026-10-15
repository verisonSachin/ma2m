"""Timeline scene director UI component: 30-frame budget sliders and motion presets."""

import gradio as gr


def render_timeline_controls():
    """Defines Gradio UI elements for timeline budgeting and camera trajectory settings."""
    with gr.Column():
        gr.Markdown("### 3. Timeline Director (30-Frame Constraint)")
        with gr.Row():
            timeline_mode = gr.Radio(
                label="Timeline Synthesis Mode",
                choices=[
                    ("Mode A: Climax Action Continuity (1 Focal Panel = 30 Frames Continuous)", "climax_continuity"),
                    ("Mode B: Multi-Panel Scene Montage (Setup 8f -> Action 14f -> Settle 8f = 30 Frames)", "multi_panel_montage"),
                ],
                value="climax_continuity",
            )
            fps_slider = gr.Slider(label="Frame Rate (FPS)", minimum=12, maximum=60, value=24, step=1)
            target_frames_box = gr.Number(label="Locked Frame Budget", value=30, interactive=False)

        with gr.Row():
            motion_preset = gr.Dropdown(
                label="Camera Motion Preset",
                choices=[
                    "Dramatic Dolly Push-In (camera_dolly_in)",
                    "Dynamic Action Dash (dynamic_action_dash)",
                    "Impact Settle & Camera Shake (impact_freeze_settle)",
                    "Horizontal Parallax Pan (parallax_pan_right)",
                    "Dutch Angle Tension Tilt (dutch_angle_tilt)",
                ],
                value="Dynamic Action Dash (dynamic_action_dash)",
            )
            motion_prompt = gr.Textbox(
                label="Motion Diffusion Prompt",
                value="slashing attack, dynamic forward motion, hair blowing in wind, intense cinematic anime action",
                lines=2,
            )

        with gr.Row():
            deflicker_toggle = gr.Checkbox(label="Enable Temporal Deflickering & Line Restoration", value=True)
            rife_interp_toggle = gr.Checkbox(label="Enable RIFE Frame Interpolation to 30fps", value=True)

        btn_render_video = gr.Button("Render Synchronized 30-Frame Video", variant="primary", size="lg")

        with gr.Row():
            video_player = gr.Video(label="Rendered 30-Frame Sequence", format="mp4", autoplay=True)
            depth_player = gr.Image(label="Dense Depth Map Preview", type="numpy")

    return {
        "timeline_mode": timeline_mode,
        "fps_slider": fps_slider,
        "target_frames_box": target_frames_box,
        "motion_preset": motion_preset,
        "motion_prompt": motion_prompt,
        "deflicker_toggle": deflicker_toggle,
        "rife_interp_toggle": rife_interp_toggle,
        "btn_render_video": btn_render_video,
        "video_player": video_player,
        "depth_player": depth_player,
    }
