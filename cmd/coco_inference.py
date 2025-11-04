import argparse
import os
import sys
from time import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processors import COCOObjectDetector

MIN_SCORE_THRESH = 0.3
TARGET_CLASS = 18  # Dog class ID

def draw_detections(detector: COCOObjectDetector, image: tf.Tensor, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray) -> None:
    """Draw bounding boxes and labels on the image."""
    print(f"Drawing detections for image with shape: {image.shape}")
    print(f"Image data range: {image.numpy().min()} to {image.numpy().max()}")

    detector.validate_image(image)

    _, ax = plt.subplots(1, figsize=(12, 8))  # More reasonable size
    ax.imshow(image)

    h, w, _ = image.shape
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    detection_count = 0
    print(f"Processing {len(boxes)} potential detections...")

    for _, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        if score >= detector.min_score_thresh:
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
            class_name = detector.COCO_CLASSES.get(int(class_id), f'Class {int(class_id)}')

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

def draw_masks(detector: COCOObjectDetector, image: tf.Tensor, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray, masks: np.ndarray) -> None:
    """Draw masks if available."""
    print(f"Drawing masks for image with shape: {image.shape}")
    print(f"Image data range: {image.numpy().min()} to {image.numpy().max()}")

    if masks is None:
        print("No masks provided, falling back to bounding boxes")
        return draw_detections(detector, image, boxes, classes, scores)

    detector.validate_image(image)

    _, ax = plt.subplots(1, figsize=(12, 8))  # Reduced size

    # Create a copy of the image for mask overlay
    image_with_masks = image.numpy().copy().astype(np.float32)

    h, w, _ = image.shape
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]

    detection_count = 0
    print(f"Processing {len(masks)} masks...")

    for _, (box, score, class_id, mask) in enumerate(zip(boxes, scores, classes, masks)):
        if score >= MIN_SCORE_THRESH:
            print(f"Processing detection {detection_count + 1} with score {score:.3f}")

            # Get class name
            class_name = detector.COCO_CLASSES.get(int(class_id), f'Class {int(class_id)}')

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

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict image class using trained model')
    parser.add_argument('image_path', help='Path to the image to predict')
    parser.add_argument('--output-dir', default='cropped_objects', help='Directory to save cropped images')

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        sys.exit(1)

    print('loading model...')
    coco_detector = COCOObjectDetector(MIN_SCORE_THRESH)
    print('model loaded!')

    image_np = coco_detector.preprocess_image(args.image_path)

    # running inference
    print('Running inference...')
    results = coco_detector.infer(image_np)
    print('Inference completed!')

    # result = {key: value.numpy() for key, value in results.items()}
    # print("Available result keys:", result.keys())

    # Extract detection results
    boxes = results.boxes
    classes = results.classes
    scores = results.scores

    print(f"Found {np.sum(scores >= MIN_SCORE_THRESH)} objects with confidence >= {MIN_SCORE_THRESH}")

    valid_detections = np.sum(scores >= MIN_SCORE_THRESH)
    print(f"Found {valid_detections} objects with confidence >= {MIN_SCORE_THRESH}")

    folder_ts = str(time())
    output_dir = os.path.join(args.output_dir, folder_ts)
    output_masked_dir = os.path.join(args.output_dir, folder_ts, "masked")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_masked_dir, exist_ok=True)

    if valid_detections > 0:
    # Draw detections with bounding boxes
        draw_detections(coco_detector, image_np[0], boxes, classes, scores)

        # Crop detected objects
        print("Cropping detected objects...")
        cropped_images = coco_detector.get_cropped_inference(image_np[0], results)

        for detection_count, cropped_image in cropped_images.items():
            # Create filename
            filename = f"{cropped_image['class_name']}_{detection_count:03d}_score_{cropped_image['score']:.2f}.jpg"
            filepath = os.path.join(output_dir, filename)

            # Convert to PIL Image and save
            pil_image = Image.fromarray(cropped_image['image'].astype(np.uint8))
            pil_image.save(filepath, 'JPEG', quality=95)

        print(f"Saved {len(cropped_images)} cropped images")

        # Handle models with masks
        print("Model supports instance segmentation masks!")
        masks = results.masks

        # Resize masks to image size
        masks_resized = []
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            if scores[i] >= MIN_SCORE_THRESH:
                # Get box coordinates
                ymin, xmin, ymax, xmax = box
                h, w = image_np.shape[1], image_np.shape[2]

                # Convert to pixel coordinates
                y1, x1 = int(ymin * h), int(xmin * w)
                y2, x2 = int(ymax * h), int(xmax * w)

                # Resize mask to box size then to full image
                mask_resized = tf.image.resize(mask[..., None], [y2-y1, x2-x1])
                mask_full = np.zeros((h, w), dtype=np.float32)
                mask_full[y1:y2, x1:x2] = mask_resized[:, :, 0].numpy()
                masks_resized.append(mask_full > 0.5)

        # Draw masks
        if masks_resized:
            valid_masks = [masks_resized[i] for i in range(len(masks_resized)) if scores[i] >= MIN_SCORE_THRESH]
            valid_boxes = boxes[scores >= MIN_SCORE_THRESH]
            valid_classes = classes[scores >= MIN_SCORE_THRESH]
            valid_scores = scores[scores >= MIN_SCORE_THRESH]

            print(type(image_np[0]), type(valid_boxes), type(valid_classes), type(valid_scores), type(valid_masks))
            draw_masks(coco_detector, image_np[0], valid_boxes, valid_classes, valid_scores, valid_masks)

            # Crop with masks
            print("Cropping with masks...")
            saved_mask_files = coco_detector.crop_detections_with_masks(
                image_np[0], valid_boxes, valid_classes, valid_scores, valid_masks,
                output_masked_dir
            )
            print(f"Saved {len(saved_mask_files)} masked cropped images")

    print("Object detection completed!")


