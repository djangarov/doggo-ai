import argparse
import os
import sys
from time import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processors import COCOObjectDetector

MIN_SCORE_THRESH = 0.3
TARGET_CLASS = 18  # Dog class ID

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect image objects using trained model')
    parser.add_argument('image_path', help='Path to the image to detect')
    parser.add_argument('--output-dir', default='cropped_objects', help='Directory to save cropped images')

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f'Error: Image file {args.image_path} not found!')
        sys.exit(1)

    print('loading model...')
    coco_detector = COCOObjectDetector(MIN_SCORE_THRESH, TARGET_CLASS)
    print('model loaded!')

    image_np = coco_detector.preprocess_image(args.image_path)
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]  # Without extension

    # running inference
    print('Running inference...')
    results = coco_detector.detect(image_np)
    print('Inference completed!')

    # Extract detection results
    boxes = results.boxes
    classes = results.classes
    scores = results.scores

    print(f'Found {np.sum(scores >= MIN_SCORE_THRESH)} objects with confidence >= {MIN_SCORE_THRESH}')

    valid_detections = np.sum(scores >= MIN_SCORE_THRESH)
    print(f'Found {valid_detections} objects with confidence >= {MIN_SCORE_THRESH}')

    folder_ts = str(time())
    output_dir = os.path.join(args.output_dir, folder_ts)
    output_masked_dir = os.path.join(args.output_dir, folder_ts, 'masked')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_masked_dir, exist_ok=True)

    if valid_detections > 0:
        print('Drawing detections with bounding boxes...')
        detection_boxes = coco_detector.get_detections_boxes(image_np[0], results)

        _, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_np[0])

        for detection_count, detection_box in detection_boxes.items():
            # Draw the bounding box
            rect = patches.Rectangle((detection_box['rectangle']['left'], detection_box['rectangle']['top']),
                                     detection_box['rectangle']['width'], detection_box['rectangle']['height'],
                                     linewidth=3, edgecolor=detection_box['color'], facecolor='none')
            ax.add_patch(rect)

            # Add label
            ax.text(rect.get_x(), rect.get_y() - 10, detection_box['label'], fontsize=12, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=detection_box['color'], alpha=0.8))

        ax.set_title(f'Object Detection Results ({len(detection_boxes)} objects found)', fontsize=16)
        ax.axis('off')
        plt.tight_layout()

        filename = f'{image_name}_{len(detection_boxes):03d}.jpg'
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
        plt.show()

        # Crop detected objects
        print('Cropping detected objects...')
        cropped_images = coco_detector.get_detections(image_np[0], results)

        for detection_count, cropped_image in cropped_images.items():
            # Create filename
            filename = f'{cropped_image['class_name']}_{detection_count:03d}_score_{cropped_image['score']:.2f}.jpg'
            filepath = os.path.join(output_dir, filename)

            # Convert to PIL Image and save
            pil_image = Image.fromarray(cropped_image['image'].astype(np.uint8))
            pil_image.save(filepath, 'JPEG', quality=95)

        print(f'Saved {len(cropped_images)} cropped images')

        # Handle models with masks
        print('Model supports instance segmentation masks!')
        masks = results.masks

        # Draw masks
        if masks is not None:
            print('Drawing detections with masks...')
            detection_masks = coco_detector.get_mask_detections_boxes(image_np[0], results)

            _, ax = plt.subplots(1, figsize=(12, 8))

            for detection_count, detection_mask in detection_masks.items():
                # Draw the bounding box
                rect = patches.Rectangle((detection_mask['rectangle']['left'], detection_mask['rectangle']['top']),
                                        detection_mask['rectangle']['width'], detection_mask['rectangle']['height'],
                                        linewidth=3, edgecolor=detection_mask['color'], facecolor='none')
                ax.add_patch(rect)

                # Add label
                ax.text(rect.get_x(), rect.get_y() - 10, detection_mask['label'], fontsize=12, color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=detection_mask['color'], alpha=0.8))

            ax.imshow(detection_mask['image'].astype(np.uint8))
            ax.set_title(f'Object Detection Results ({len(detection_masks)} objects found)', fontsize=16)
            ax.axis('off')
            plt.tight_layout()

            filename = f'{image_name}_{len(detection_masks):03d}_mask.jpg'
            filepath = os.path.join(output_masked_dir, filename)

            plt.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
            plt.show()

            # Crop with masks
            print('Cropping with masks...')
            cropped_mask_images = coco_detector.get_mask_detections(image_np[0], results)

            for detection_count, cropped_image in cropped_mask_images.items():
                # Create filename
                filename = f'{cropped_image['class_name']}_{detection_count:03d}_score_{cropped_image['score']:.2f}.jpg'
                filepath = os.path.join(output_masked_dir, filename)

                # Convert to PIL Image and save
                pil_image = Image.fromarray(cropped_image['image'].astype(np.uint8))
                pil_image.save(filepath, 'JPEG', quality=95)

            print(f'Saved {len(cropped_mask_images)} masked cropped images')

    print('Object detection completed!')


