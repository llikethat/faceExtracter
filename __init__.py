"""
Face Extractor for ComfyUI
Version 4.0 - Generic Face Extraction Tool

Features:
- Memory-aware processing with GPU flush
- Aspect ratio preservation
- Multiple detection backends with different licenses
- Commercial-friendly options available

LICENSE NOTICE:
This tool supports multiple face detection backends with different licenses:
- InsightFace: NON-COMMERCIAL USE ONLY (see insightface license)
- MediaPipe: Apache 2.0 (Commercial OK)
- OpenCV Cascade: BSD (Commercial OK)

For commercial projects, use mediapipe or opencv_cascade backend!
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
