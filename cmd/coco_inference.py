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

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict image class using trained model')
    parser.add_argument('image_path', help='Path to the image to predict')
    parser.add_argument('--crop', action='store_true', help='Crop detected objects and save them')
    parser.add_argument('--crop-class', type=int, help='Only crop objects of specific class (e.g., 18 for dogs)')
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

    result = {key: value.numpy() for key, value in results.items()}
    print("Available result keys:", result.keys())

    # Extract detection results
    boxes = result['detection_boxes'][0]
    classes = result['detection_classes'][0]
    scores = result['detection_scores'][0]

    print(f"Found {np.sum(scores >= MIN_SCORE_THRESH)} objects with confidence >= {MIN_SCORE_THRESH}")

    valid_detections = np.sum(scores >= MIN_SCORE_THRESH)
    print(f"Found {valid_detections} objects with confidence >= {MIN_SCORE_THRESH}")

    folder_ts = str(time())
    output_dir = os.path.join(args.output_dir, folder_ts)
    output_masked_dir = os.path.join(args.output_dir, folder_ts, "masked")

    if valid_detections > 0:
    # Draw detections with bounding boxes
        coco_detector.draw_detections(image_np[0], boxes, classes, scores, args.crop_class)

        # Crop detected objects if requested
        if args.crop:
            print("Cropping detected objects...")
            saved_files = coco_detector.crop_detections(image_np[0], boxes, classes, scores,
                            output_dir, args.crop_class)
            print(f"Saved {len(saved_files)} cropped images")

        # Handle models with masks
        if 'detection_masks' in result:
            print("Model supports instance segmentation masks!")
            masks = result['detection_masks'][0]

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
                coco_detector.draw_masks(image_np[0], valid_boxes, valid_classes, valid_scores, valid_masks, args.crop_class)

                # Crop with masks if requested
                if args.crop:
                    print("Cropping with masks...")
                    saved_mask_files = coco_detector.crop_detections_with_masks(
                        image_np[0], valid_boxes, valid_classes, valid_scores, valid_masks,
                        output_masked_dir, args.crop_class
                    )
                    print(f"Saved {len(saved_mask_files)} masked cropped images")

    print("Object detection completed!")


