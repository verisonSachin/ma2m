"""Ingestion package: panel extraction, speech bubble detection, and inpainting."""

from src.ingestion.panel_extractor import Panel, PanelExtractor
from src.ingestion.bubble_detector import BubbleDetector, BubbleRegion
from src.ingestion.inpainter import MangaInpainter

__all__ = [
    "Panel",
    "PanelExtractor",
    "BubbleDetector",
    "BubbleRegion",
    "MangaInpainter",
]
