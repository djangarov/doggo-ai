from processors.coco_object_detector import COCOObjectDetector
from processors.image_classifier import ImageClassifier


class ClassificationService:
    def __init__(self, detector: COCOObjectDetector, classifier: ImageClassifier) -> None:
        self.detector = detector
        self.classifier = classifier