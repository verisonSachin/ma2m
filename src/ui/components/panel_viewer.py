"""Panel viewer UI component: renders annotated manga pages and extracted panel thumbnails."""

from typing import List, Tuple
import gradio as gr
import numpy as np
from PIL import Image

from src.ingestion.panel_extractor import Panel, PanelExtractor


def render_panel_gallery():
    """Defines Gradio UI elements for manga panel inspection."""
    with gr.Column():
        gr.Markdown("### 1. Ingestion & Japanese RTL Reading Order")
        with gr.Row():
            page_uploads = gr.File(
                label="Upload ~3 Consecutive Manga Pages (PNG/JPG)",
                file_count="multiple",
                file_types=["image"],
            )
            btn_extract = gr.Button("Extract & Sequence Panels", variant="primary")

        with gr.Row():
            annotated_page_view = gr.Image(label="Annotated Page with RTL Reading Order", type="numpy")
            panel_gallery = gr.Gallery(
                label="Sequenced Panels (Ordered Right-to-Left, Top-to-Bottom)",
                columns=3,
                rows=2,
                height=450,
                object_fit="contain",
            )

        with gr.Row():
            inpaint_toggle = gr.Checkbox(label="Enable Text Bubble & SFX Inpainting", value=True)
            inpainted_preview = gr.Image(label="Clean Inpainted Panel Preview", type="numpy")

    return {
        "page_uploads": page_uploads,
        "btn_extract": btn_extract,
        "annotated_page_view": annotated_page_view,
        "panel_gallery": panel_gallery,
        "inpaint_toggle": inpaint_toggle,
        "inpainted_preview": inpainted_preview,
    }
