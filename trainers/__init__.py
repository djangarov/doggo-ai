"""
Trainers package.

Contains training modules for object detection.
"""

__version__ = "1.0.0"

from .base_trainer import BaseTrainer
from .custom_trainer import CustomTrainer
from .efficientnet_b7_trainer import EfficientNetB7Trainer
from .inception_resnet_v2_trainer import InceptionResNetV2Trainer
from .inception_v3_trainer import InceptionV3Trainer
from .resnet50_trainer import ResNet50Trainer
from .vgg19_trainer import VGG19Trainer
from .model_factory import ModelFactory

__all__ = [
    "BaseTrainer",
    "CustomTrainer",
    "EfficientNetB7Trainer",
    "InceptionResNetV2Trainer",
    "InceptionV3Trainer",
    "ResNet50Trainer",
    "VGG19Trainer",
    "ModelFactory",
]

def list_trainers():
    """List available trainer modules."""
    return [
        "BaseTrainer",
        "CustomTrainer",
        "EfficientNetB7Trainer",
        "InceptionResNetV2Trainer",
        "InceptionV3Trainer",
        "ResNet50Trainer",
        "VGG19Trainer",
        "ModelFactory",
    ]