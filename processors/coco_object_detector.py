import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub

from processors import ImageProcessor

MODEL_HANDLE = "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"

class COCOObjectDetector(ImageProcessor):
    __COCO_CLASSES = {
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

    def __init__(self, min_score_thresh: float = 0.3):
        self.model = hub.load(MODEL_HANDLE)
        self.min_score_thresh = min_score_thresh

    def infer(self, image: tf.Tensor) -> dict:
        """Run inference on the image and return detection results."""
        outputs = self.model(image)

        return outputs

    def validate_image(self, image: tf.Tensor) -> bool:
        """Validate the image data."""
        if image is None or image.numpy().size == 0:
            raise ValueError("Invalid image data")

        if image.numpy().max() == image.numpy().min():
            raise ValueError(f"Image appears to be uniform (all pixels have value {image.max()})")

        return True

    def crop_detections(self, image: tf.Tensor, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray,
                    output_dir: str = "cropped_objects", target_class: int = None) -> list:
        """
        Crop detected objects from the original image and save them as separate files.

        Args:
            image: Original image as numpy array
            boxes: Detection bounding boxes (normalized coordinates)
            classes: Detection class IDs
            scores: Detection confidence scores
            output_dir: Directory to save cropped images
            target_class: If specified, only crop objects of this class (e.g., 18 for dogs)

        Returns:
            List of saved file paths
        """
        self.validate_image(image)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        h, w, _ = image.shape
        saved_files = []
        detection_count = 0

        print(f"Cropping detections from image with shape: {image.shape}")

        for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
            # Filter by confidence and optionally by class
            if score >= self.min_score_thresh:
                if target_class is not None and int(class_id) != target_class:
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

                # Get class name for filename
                class_name = self.__COCO_CLASSES.get(int(class_id), f'class_{int(class_id)}')

                # Create filename
                filename = f"{class_name}_{detection_count:03d}_score_{score:.2f}.jpg"
                filepath = os.path.join(output_dir, filename)

                # Convert to PIL Image and save
                pil_image = Image.fromarray(cropped_image.astype(np.uint8))
                pil_image.save(filepath, 'JPEG', quality=95)

                saved_files.append(filepath)
                print(f"Saved detection {detection_count + 1}: {filename} ({cropped_image.shape[1]}x{cropped_image.shape[0]})")
                print(f"  Original box: ({left}, {top}) -> ({right}, {bottom})")

                detection_count += 1

        print(f"Cropped and saved {len(saved_files)} objects to '{output_dir}'")
        return saved_files

    def crop_detections_with_masks(self, image: tf.Tensor, boxes: np.ndarray, classes: np.ndarray,
                                scores: np.ndarray, masks: np.ndarray, output_dir: str = "cropped_objects_masked",
                                target_class: int = None) -> list:
        """
        Crop detected objects using masks for precise extraction.

        Args:
            image: Original image as numpy array
            boxes: Detection bounding boxes (normalized coordinates)
            classes: Detection class IDs
            scores: Detection confidence scores
            masks: Detection masks
            output_dir: Directory to save cropped images
            target_class: If specified, only crop objects of this class

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
                if target_class is not None and int(class_id) != target_class:
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
                class_name = self.__COCO_CLASSES.get(int(class_id), f'class_{int(class_id)}')

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

    def draw_detections(self, image: tf.Tensor, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray, target_class: int = None) -> None:
        """Draw bounding boxes and labels on the image."""
        print(f"Drawing detections for image with shape: {image.shape}")
        print(f"Image data range: {image.numpy().min()} to {image.numpy().max()}")

        self.validate_image(image)

        _, ax = plt.subplots(1, figsize=(12, 8))  # More reasonable size
        ax.imshow(image)

        h, w, _ = image.shape
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        detection_count = 0
        print(f"Processing {len(boxes)} potential detections...")

        for _, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
            if score >= self.min_score_thresh:
                if target_class is not None and int(class_id) != target_class:
                    continue

                ymin, xmin, ymax, xmax = box

                # Convert normalized coordinates to pixel coordinates
                left, top = int(xmin * w), int(ymin * h)
                width, height = int((xmax - xmin) * w), int((ymax - ymin) * h)

                color = colors[detection_count % len(colors)]

                # Create rectangle patch
                rect = patches.Rectangle((left, top), width, height,
                                    linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                # Get class name
                class_name = self.__COCO_CLASSES.get(int(class_id), f'Class {int(class_id)}')

                # Add label with background
                label = f'{class_name}: {score:.2f}'
                ax.text(left, top - 10, label, fontsize=12, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

                print(f"Detection {detection_count + 1}: {class_name} (Class {int(class_id)}), Score: {score:.3f}")
                print(f"  Box: ({left}, {top}) -> ({left+width}, {top+height})")
                detection_count += 1

        ax.set_title(f'Object Detection Results ({detection_count} objects found)', fontsize=16)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def draw_masks(self, image: tf.Tensor, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray, masks: np.ndarray, target_class: int = None) -> None:
        """Draw masks if available."""
        print(f"Drawing masks for image with shape: {image.shape}")
        print(f"Image data range: {image.numpy().min()} to {image.numpy().max()}")

        if masks is None:
            print("No masks provided, falling back to bounding boxes")
            return self.draw_detections(image, boxes, classes, scores, target_class)

        self.validate_image(image)

        _, ax = plt.subplots(1, figsize=(12, 8))  # Reduced size

        # Create a copy of the image for mask overlay
        image_with_masks = image.numpy().copy().astype(np.float32)

        h, w, _ = image.shape
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]

        detection_count = 0
        print(f"Processing {len(masks)} masks...")

        for _, (box, score, class_id, mask) in enumerate(zip(boxes, scores, classes, masks)):
            if score >= self.min_score_thresh:
                if target_class is not None and int(class_id) != target_class:
                    continue

                print(f"Processing detection {detection_count + 1} with score {score:.3f}")

                # Get class name
                class_name = self.__COCO_CLASSES.get(int(class_id), f'Class {int(class_id)}')

                # Apply mask overlay
                color = colors[detection_count % len(colors)]

                # Ensure mask is 2D
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]  # Take first channel if 3D

                # Create colored mask
                mask_colored = np.zeros_like(image_with_masks)
                for c in range(3):
                    mask_colored[:, :, c] = mask * color[c]

                # Blend mask with image (only where mask > 0.5)
                mask_binary = mask > 0.5
                image_with_masks = np.where(mask_binary[..., None],
                                        0.7 * image_with_masks + 0.3 * mask_colored,
                                        image_with_masks)

                # Draw bounding box
                ymin, xmin, ymax, xmax = box
                left, top = int(xmin * w), int(ymin * h)
                width, height = int((xmax - xmin) * w), int((ymax - ymin) * h)

                rect = patches.Rectangle((left, top), width, height,
                                    linewidth=3, edgecolor=np.array(color)/255, facecolor='none')
                ax.add_patch(rect)

                # Add label
                label = f'{class_name}: {score:.2f}'
                ax.text(left, top - 10, label, fontsize=12, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=np.array(color)/255, alpha=0.8))

                print(f"Detection {detection_count + 1}: {class_name} (Class {int(class_id)}), Score: {score:.3f}")
                print(f"  Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
                detection_count += 1

        ax.imshow(image_with_masks.astype(np.uint8))
        ax.set_title(f'Object Detection with Masks ({detection_count} objects found)', fontsize=16)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
