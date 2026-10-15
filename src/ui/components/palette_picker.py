"""Color palette manager UI component: HEX swatches, ref upload, and K-Means extraction."""

import gradio as gr
from src.colorization.palette_manager import CharacterPalette


def render_palette_controls():
    """Defines Gradio UI elements for character color reference locking."""
    with gr.Column():
        gr.Markdown("### 2. Character Reference & Color Palette Locking")
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                label="Palette Preset",
                choices=["Protagonist (Heroic Red/Navy)", "Antagonist (Shadow/Obsidian)", "Companion (Mystic Violet)", "Custom"],
                value="Protagonist (Heroic Red/Navy)",
            )
            ref_image_upload = gr.Image(label="Upload Character Reference Sheet / Colored Art", type="numpy")
            btn_extract_palette = gr.Button("Extract Palette from Image", variant="secondary")

        gr.Markdown("#### Key Entity Color Anchors (HEX)")
        with gr.Row():
            hair_color = gr.ColorPicker(label="Hair Color", value="#C0392B")
            skin_color = gr.ColorPicker(label="Skin Tone", value="#F5CBA7")
            eyes_color = gr.ColorPicker(label="Eye Color", value="#2980B9")
            outfit_color = gr.ColorPicker(label="Primary Outfit", value="#1B2631")
            accent_color = gr.ColorPicker(label="Accent Detail", value="#F39C12")

        with gr.Row():
            btn_preview_color = gr.Button("Generate Colorized Keyframe Preview", variant="primary")
            colorized_preview = gr.Image(label="Colorized Panel Preview", type="numpy")

    return {
        "preset_dropdown": preset_dropdown,
        "ref_image_upload": ref_image_upload,
        "btn_extract_palette": btn_extract_palette,
        "hair_color": hair_color,
        "skin_color": skin_color,
        "eyes_color": eyes_color,
        "outfit_color": outfit_color,
        "accent_color": accent_color,
        "btn_preview_color": btn_preview_color,
        "colorized_preview": colorized_preview,
    }
