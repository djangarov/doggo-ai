import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from processors import ImageProcessor

MODEL_URL = 'https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1'
PROJECT_ROOT = Path(__file__).parent.parent
STORAGE_DIR = f'{PROJECT_ROOT}/storage/models/'
MODEL_NAME = 'mask_rcnn_inception_resnet_v2_1024x1024'

class DetectionResult:
    def __init__(self,
                 boxes: tf.Tensor,
                 classes: tf.Tensor,
                 scores: tf.Tensor,
                 masks: tf.Tensor | None) -> None:
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.masks = masks


class COCOObjectDetector(ImageProcessor):
    COCO_CLASSES = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
        48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
        53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
        58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
        63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
        76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
        85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush'
    }
    BOXES_COLORS = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'brown': (165, 42, 42),
        'pink': (255, 192, 203),
        'gray': (128, 128, 128),
        'olive': (128, 128, 0),
        'cyan': (0, 255, 255)
    }

    def __init__(self,
                 min_score_thresh: float,
                 target_class: int,
                 model_url: str = MODEL_URL,
                 storage_dir: str = STORAGE_DIR,
                 model_name: str = MODEL_NAME) -> None:
        self.min_score_thresh = min_score_thresh
        self.target_class = target_class
        self.model_url = model_url
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.model = self.__load_model()

    def __load_model(self) -> hub.KerasLayer:
        """
        Load the pre-trained Mask R-CNN model from TensorFlow Hub.
        """
        path = f'{self.storage_dir}{self.model_name}'

        try:
            if os.path.exists(path):
                model = hub.load(path)
                return model

            model = hub.load(self.model_url)
            tf.saved_model.save(model, path)

            return model
        except Exception as e:
            print(f'Error loading model: {e}')
            raise

    def detect(self, image: tf.Tensor) -> DetectionResult:
        """
        Run inference on the image and return detection results.

        Args:
            image: Input image tensor

        Returns:
            DetectionResult object containing boxes, classes, scores, and masks.
        """
        result = self.model(image)

        return DetectionResult(
            boxes=result['detection_boxes'][0],
            classes=result['detection_classes'][0],
            scores=result['detection_scores'][0],
            masks=result['detection_masks'][0]
        )

    def validate_image(self, image: tf.Tensor) -> bool:
        """
        Validate the image data.

        Returns:
            bool: True if the image is valid, raises ValueError otherwise.
        """
        if image is None or image.numpy().size == 0:
            raise ValueError('Invalid image data')

        if image.numpy().max() == image.numpy().min():
            raise ValueError(f'Image appears to be uniform (all pixels have value {image.max()})')

        return True

    def get_detections_boxes(self, image: tf.Tensor, detection: DetectionResult) -> dict:
        """
        Draw bounding boxes and labels on the image.

        Args:
            image: Input image tensor
            detection: Detection results from model inference

        Returns:
            Dictionary with detection IDs as keys and box details as values.
        """
        self.validate_image(image)

        h, w, _ = image.shape
        boxes = {}
        detection_count = 0

        for _, (box, score, class_id) in enumerate(zip(detection.boxes, detection.scores, detection.classes)):
            if score < self.min_score_thresh:
                continue

            ymin, xmin, ymax, xmax = box

            # Convert normalized coordinates to pixel coordinates
            left, top = int(xmin * w), int(ymin * h)
            width, height = int((xmax - xmin) * w), int((ymax - ymin) * h)

            color_names = list(self.BOXES_COLORS.keys())
            color_name = color_names[detection_count % len(color_names)]

            # Get class name
            class_name = self.COCO_CLASSES.get(int(class_id), f'Class {int(class_id)}')

            # Add label
            label = f'{class_name}: {score:.2f}'

            boxes[detection_count] = {
                'color': color_name,
                'rectangle': {
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height
                },
                'label': label
            }
            detection_count += 1

        return boxes

    def get_mask_detections_boxes(self, image: tf.Tensor, detection: DetectionResult) -> dict:
        """
        Draw masks if available.

        Returns:
            dict: A dictionary with detection IDs as keys and mask details as values.
        """
        if detection.masks is None:
            print('No masks provided, falling back to bounding boxes')
            return self.get_detections_boxes(image, detection)

        self.validate_image(image)

        # Create a copy of the image for mask overlay
        image_with_masks = image.numpy().copy().astype(np.float32)

        h, w, _ = image.shape
        boxes = {}
        detection_count = 0

        for _, (box, score, class_id, mask) in enumerate(zip(detection.boxes, detection.scores, detection.classes, detection.masks)):
            if score < self.min_score_thresh:
                continue

            # Get class name
            class_name = self.COCO_CLASSES.get(int(class_id), f'Class {int(class_id)}')

            # Get RGB values directly without color names
            color_names = list(self.BOXES_COLORS.keys())
            color_name = color_names[detection_count % len(color_names)]
            color_rgb = self.BOXES_COLORS[color_name]

            ymin, xmin, ymax, xmax = box

            # Convert to pixel coordinates
            y1, x1 = int(ymin * h), int(xmin * w)
            y2, x2 = int(ymax * h), int(xmax * w)

            # Resize mask to box size then to full image
            mask_resized = tf.image.resize(mask[..., None], [y2-y1, x2-x1])
            mask_full = np.zeros((h, w), dtype=np.float32)
            mask_full[y1:y2, x1:x2] = mask_resized[:, :, 0].numpy()

            # Ensure mask is 2D
            if len(mask.shape) == 3:
                mask_full = mask_full[:, :, 0]  # Take first channel if 3D

            # Create colored mask
            mask_colored = np.zeros_like(image_with_masks)
            for c in range(3):
                mask_colored[:, :, c] = mask_full * color_rgb[c]

            # Blend mask with image (only where mask > 0.5)
            mask_binary = mask_full > 0.5
            image_with_masks = np.where(mask_binary[..., None],
                                    0.7 * image_with_masks + 0.3 * mask_colored,
                                    image_with_masks)

            # Draw bounding box
            ymin, xmin, ymax, xmax = box
            left, top = int(xmin * w), int(ymin * h)
            width, height = int((xmax - xmin) * w), int((ymax - ymin) * h)

            # Add label
            label = f'{class_name}: {score:.2f}'

            boxes[detection_count] = {
                'image': image_with_masks,
                'color': color_name,
                'rectangle': {
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height
                },
                'label': label
            }
            detection_count += 1

        return boxes

    def get_detections(self, image: tf.Tensor, detection: DetectionResult) -> dict:
        """
        Crop detected objects from the original image.

        Returns:
            dict: A dictionary with detection IDs as keys and cropped image data as values.
        """
        self.validate_image(image)

        h, w, _ = image.shape
        detection_count = 0
        cropped_images = {}

        for i, (box, score, class_id) in enumerate(zip(detection.boxes, detection.scores, detection.classes)):
            # Filter by confidence and optionally by class
            if score < self.min_score_thresh:
                continue

            if int(class_id) != self.target_class:
                continue

            ymin, xmin, ymax, xmax = box

            # Convert normalized coordinates to pixel coordinates
            left = int(xmin * w)
            top = int(ymin * h)
            right = int(xmax * w)
            bottom = int(ymax * h)

            # Ensure coordinates are within image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            # Skip if bounding box is too small
            if (right - left) < 10 or (bottom - top) < 10:
                print(f'Skipping detection {i}: bounding box too small')
                continue

            # Crop the image
            cropped_image = image.numpy()[top:bottom, left:right]

            class_name = self.COCO_CLASSES.get(int(class_id), f'class_{int(class_id)}')

            cropped_images[detection_count] = {
                'image': cropped_image,
                'class_id': class_id,
                'score': score,
                'class_name': class_name
            }

            detection_count += 1

        return cropped_images

    def get_mask_detections(self, image: tf.Tensor, detection: DetectionResult) -> dict:
        """
        Crop detected objects using masks for precise extraction.

        Returns:
             dict: A dictionary with detection IDs as keys and cropped image data as values.
        """
        self.validate_image(image)

        h, w, _ = image.shape
        detection_count = 0
        cropped_images = {}

        for i, (box, score, class_id, mask) in enumerate(zip(detection.boxes, detection.scores, detection.classes, detection.masks)):
            # Filter by confidence and optionally by class
            if score < self.min_score_thresh:
                continue

            if int(class_id) != self.target_class:
                continue

            ymin, xmin, ymax, xmax = box

            # Convert normalized coordinates to pixel coordinates
            left = int(xmin * w)
            top = int(ymin * h)
            right = int(xmax * w)
            bottom = int(ymax * h)

             # Calculate 5% padding
            bbox_width = right - left
            bbox_height = bottom - top
            padding_x = int(bbox_width * 0.05)
            padding_y = int(bbox_height * 0.05)

            # Apply padding
            left_padded = left - padding_x
            top_padded = top - padding_y
            right_padded = right + padding_x
            bottom_padded = bottom + padding_y

            # Ensure coordinates are within image bounds
            left = max(0, left_padded)
            top = max(0, top_padded)
            right = min(w, right_padded)
            bottom = min(h, bottom_padded)

            # Skip if bounding box is too small
            if (right - left) < 10 or (bottom - top) < 10:
                print(f'Skipping detection {i}: bounding box too small')
                continue

            # Ensure mask is 2D
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            # Convert mask to float32 for TensorFlow resize
            mask_float = tf.cast(mask, tf.float32)

            # Resize mask to match bounding box
            mask_resized = tf.image.resize(mask_float[..., None], [bottom-top, right-left])
            mask_binary = mask_resized[:, :, 0].numpy() > 0.5

            # Crop the image
            cropped_image = image.numpy()[top:bottom, left:right].copy()

            # Apply mask (set background to white or transparent)
            cropped_image = cropped_image.copy()
            cropped_image[~mask_binary] = [255, 255, 255]  # White background

            # Get class name for filename
            class_name = self.COCO_CLASSES.get(int(class_id), f'class_{int(class_id)}')

            cropped_images[detection_count] = {
                'image': cropped_image,
                'class_id': class_id,
                'score': score,
                'class_name': class_name
            }

            detection_count += 1

        return cropped_images

