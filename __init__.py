"""
DFL Face Extractor for ComfyUI
Version 3.0 - Memory-aware chunked processing

Automatically extracts faces from video/images matching a reference face.
Designed for DeepFaceLab de-aging and face swap workflows.

Key Features:
- Built-in video loading (no VHS dependency needed for large videos)
- Memory-aware chunked processing
- Streaming mode for unlimited video length
- GPU acceleration with automatic memory management
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./js"
