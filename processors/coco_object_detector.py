import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub

from processors import ImageProcessor

MODEL_HANDLE = "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"

class InferenceResult:
    def __init__(self, boxes, classes, scores, masks):
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

    def __init__(self, min_score_thresh: float = 0.3, target_class: int = 18):
        self.model = hub.load(MODEL_HANDLE)
        self.min_score_thresh = min_score_thresh
        self.target_class = target_class

    def infer(self, image: tf.Tensor) -> dict:
        """Run inference on the image and return detection results."""
        result = self.model(image)

        return InferenceResult(
            boxes=result['detection_boxes'][0],
            classes=result['detection_classes'][0],
            scores=result['detection_scores'][0],
            masks=result['detection_masks'][0]
        )

    def validate_image(self, image: tf.Tensor) -> bool:
        """Validate the image data."""
        if image is None or image.numpy().size == 0:
            raise ValueError("Invalid image data")

        if image.numpy().max() == image.numpy().min():
            raise ValueError(f"Image appears to be uniform (all pixels have value {image.max()})")

        return True

    def get_cropped_inference(self, image: tf.Tensor, inference_result: InferenceResult) -> dict:
        """
        Crop detected objects from the original image.

        Returns:
            dict: A dictionary with detection IDs as keys and cropped image data as values.
        """
        self.validate_image(image)

        h, w, _ = image.shape
        detection_count = 0
        cropped_images = {}

        for i, (box, score, class_id) in enumerate(zip(inference_result.boxes, inference_result.scores, inference_result.classes)):
            # Filter by confidence and optionally by class
            if score >= self.min_score_thresh:
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
                    print(f"Skipping detection {i}: bounding box too small")
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

    def crop_detections_with_masks(self, image: tf.Tensor, boxes: np.ndarray, classes: np.ndarray,
                                scores: np.ndarray, masks: np.ndarray, output_dir: str = "cropped_objects_masked") -> list:
        """
        Crop detected objects using masks for precise extraction.

        Returns:
            List of saved file paths
        """
        self.validate_image(image)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        h, w, _ = image.shape
        saved_files = []
        detection_count = 0

        print(f"Cropping detections with masks from image with shape: {image.shape}")

        for i, (box, score, class_id, mask) in enumerate(zip(boxes, scores, classes, masks)):
            # Filter by confidence and optionally by class
            if score >= self.min_score_thresh:
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
                cropped_image_masked = cropped_image.copy()
                cropped_image_masked[~mask_binary] = [255, 255, 255]  # White background

                # Get class name for filename
                class_name = self.COCO_CLASSES.get(int(class_id), f'class_{int(class_id)}')

                # Masked crop
                filename_masked = f"{class_name}_{detection_count:03d}_masked_score_{score:.2f}.jpg"
                filepath_masked = os.path.join(output_dir, filename_masked)
                pil_image_masked = Image.fromarray(cropped_image_masked.astype(np.uint8))
                pil_image_masked.save(filepath_masked, 'JPEG', quality=95)

                saved_files.append(filepath_masked)
                print(f"Saved detection {detection_count + 1}: {filename_masked}")
                print(f"  Size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")

                detection_count += 1

        print(f"Cropped and saved {len(saved_files)} images to '{output_dir}'")
        return saved_files

