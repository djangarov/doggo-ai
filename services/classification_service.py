import os

from matplotlib import patches, pyplot as plt
import numpy as np
import tensorflow as tf
from processors.coco_object_detector import COCOObjectDetector
from processors.image_classifier import ImageClassifier


class ClassificationService:
    def __init__(self, detector: COCOObjectDetector, classifier: ImageClassifier) -> None:
        self.detector = detector
        self.classifier = classifier

    def detect_objects(self, image_path: str) -> tuple|None:
        # Preprocessing the image
        print(f'Preprocessing image from {image_path}...')
        tensor_image = self.detector.preprocess_image_from_file(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Without extension

        # Running inference
        print('Running inference...')
        detected_objects = self.detector.detect(tensor_image)
        print('Inference completed!')

        # Analyzing results
        valid_detections = np.sum(detected_objects.scores >= self.detector.min_score_thresh)
        print(f'Found {valid_detections} objects with confidence >= {self.detector.min_score_thresh}')

        if valid_detections == 0:
            print('No objects detected with sufficient confidence. Exiting.')
            return

        return tensor_image[0], image_name, detected_objects

    def get_detections_boxes(self, tensor_image: tf.Tensor, detected_objects: dict) -> dict:
        return self.detector.get_detections_boxes(tensor_image, detected_objects)

    def get_mask_detections_boxes(self, tensor_image: tf.Tensor, detected_objects: dict) -> dict:
        return self.detector.get_mask_detections_boxes(tensor_image, detected_objects)

    def draw_and_save_detection_boxes(self, image_name: str, tensor_image: tf.Tensor, detection_boxes: dict, output_dir: str):
        """Draw detection boxes around the detected objects on the image and save the result."""
        _, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(tensor_image[0])

        composite_image = None

        for detection_box in detection_boxes.values():
            # Draw the bounding box
            rect = patches.Rectangle((detection_box['rectangle']['left'], detection_box['rectangle']['top']),
                                        detection_box['rectangle']['width'], detection_box['rectangle']['height'],
                                        linewidth=3, edgecolor=detection_box['color'], facecolor='none')
            ax.add_patch(rect)

            # Add label
            ax.text(rect.get_x(), rect.get_y() - 10, detection_box['label'], fontsize=12, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=detection_box['color'], alpha=0.8))

            if 'image' in detection_box:
                composite_image = detection_box['image']

        # Display the composite image
        if composite_image is not None:
            ax.imshow(composite_image.astype(np.uint8))

        ax.set_title(f'Object Detection Results ({len(detection_boxes)} objects found)', fontsize=16)
        ax.axis('off')
        plt.tight_layout()

        filename = f'{image_name}_{len(detection_boxes):03d}.jpg'
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, format='jpg', dpi=300, bbox_inches='tight')

    def classify_image(self, image_path: str):
        pass