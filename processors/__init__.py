
"""
Processors package.

Contains processors module
"""

__version__ = "1.0.0"

from .image_processor import ImageProcessor
from .image_predictor import ImagePredictor
from .coco_object_detector import COCOObjectDetector

__all__ = [
    "ImageProcessor",
    "ImagePredictor",
    "COCOObjectDetector",
]

def list_processors():
    """List available processors modules."""
    return [
        "ImageProcessor",
        "ImagePredictor",
        "COCOObjectDetector",
    ]