import os
import sys
import argparse
import json
import re
from time import time
from dotenv import load_dotenv

import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llms import OllamaClient, GemClient, ClientInterface
from llms.prompts import OWNER_FOCUSED_BREED_DETAILS
from processors import COCOObjectDetector, ImageClassifier

load_dotenv()

MIN_SCORE_THRESH = 0.3
TARGET_CLASS = 18  # Dog class ID

def draw_and_save_detection_boxes(image_name: str, tensor_image: tf.Tensor, detection_boxes: dict, output_dir: str):
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

def proceed_predictions(cropped_images: dict, classifier: ImageClassifier, output_dir: str) -> list[dict]:
    predictions_result = []

    for detection_count, cropped_image in cropped_images.items():
        # Create filename
        filename = f"{cropped_image['class_name']}_{detection_count:03d}_score_{cropped_image['score']:.2f}"
        filepath = os.path.join(output_dir, f"{filename}.jpg")

        # Convert to PIL Image and save
        pil_image = Image.fromarray(cropped_image['image'].astype(np.uint8))
        pil_image.save(filepath, 'JPEG', quality=95)

        processed_image = classifier.preprocess_image(pil_image, classifier.model.input_shape[1:3])

        # Make prediction
        print('Making prediction...')
        prediction_result = classifier.predict_image(processed_image)

        # Get top 5 predictions
        top_5_predictions = prediction_result.get_top(5)

        prediction_data = {
            'image_path': filepath,
            'predicted_class_id': int(prediction_result.predicted_class),  # Convert to Python int
            'confidence': float(prediction_result.confidence),
            'predicted_class_name': (prediction_result.class_names[prediction_result.predicted_class]
                                if prediction_result.class_names and
                                prediction_result.predicted_class < len(prediction_result.class_names)
                                else f'Class {prediction_result.predicted_class}'),
            'top_5_predictions': [],
            'timestamp': float(time())  # Ensure this is Python float
        }

        print('\n' + '='*50)
        print(f'PREDICTION RESULTS')
        print('='*50)
        print(f'Predicted class ID: {prediction_result.predicted_class}')
        print(f'Confidence: {prediction_result.confidence:.4f} ({prediction_result.confidence*100:.2f}%)')

        print(f'Predicted class name: {prediction_result.class_names[prediction_result.predicted_class]}')
        print('\nTop 5 predictions:')
        for i, (idx, conf) in enumerate(top_5_predictions):
            if idx < len(prediction_result.class_names):
                print(f'{i+1}. {prediction_result.class_names[idx]}: {conf:.4f} ({conf*100:.2f}%)')
                prediction_data['top_5_predictions'].append({
                    'rank': i + 1,
                    'class_id': int(idx),
                    'class_name': prediction_result.class_names[idx],
                    'confidence': float(conf),
                    'confidence_percentage': float(conf * 100)
                })

        # Save to JSON file
        json_filename = f"{filename}.json"
        json_filepath = os.path.join(output_dir, json_filename)

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, indent=2)

        print(f'Predictions saved to: {json_filepath}')
        predictions_result.append(prediction_data)

    print(f'Saved {len(cropped_images)} cropped images')

    return predictions_result

def get_info_for_prediction(predictions: list[dict], predictions_masked: list[dict] | None, llm_client: ClientInterface) -> None:
    asked_for = []

    for idx, prediction in enumerate(predictions):
        top_prediction = prediction['predicted_class_name']

        if predictions_masked is not None:
            top_prediction = prediction['predicted_class_name'] if prediction['confidence'] >= predictions_masked[idx]['confidence'] else predictions_masked[idx]['predicted_class_name']

        top_class_name = re.sub(r'[-_]', ' ', re.sub(r'^[^-]*-', '', top_prediction))

        if top_class_name in asked_for:
            continue  # Skip if already asked for this class

        asked_for.append(top_class_name)
        print('\n' + '='*50)
        print(f'Asking LLM ({llm_client.__class__.__name__}) for details about: {top_class_name}')
        print('\n' + '='*50)
        question = OWNER_FOCUSED_BREED_DETAILS.format(breed=top_class_name)
        llm_client.stream_chat(question)


def main():
    parser = argparse.ArgumentParser(description='Classify image class using trained model')
    parser.add_argument('model_path', help='Path to the trained model file (.keras)')
    parser.add_argument('image_path', help='Path to the image to classify')
    parser.add_argument('dataset', help='Path to dataset directory for class names')
    parser.add_argument('output_dir', default='cropped_objects', help='Directory to save cropped images')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f'Error: Model file {args.model_path} not found!')
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f'Error: Image file {args.image_path} not found!')
        sys.exit(1)

    if not os.path.exists(args.dataset):
        print(f'Warning: Dataset directory {args.dataset} not found!')
        sys.exit(1)

    print('Loading model...')
    coco_detector = COCOObjectDetector(MIN_SCORE_THRESH, TARGET_CLASS)
    print('Model loaded!')

    tensor_image = coco_detector.preprocess_image_from_file(args.image_path)
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]  # Without extension

    # Running inference
    print('Running inference...')
    results = coco_detector.detect(tensor_image)
    print('Inference completed!')

    valid_detections = np.sum(results.scores >= MIN_SCORE_THRESH)
    print(f'Found {valid_detections} objects with confidence >= {MIN_SCORE_THRESH}')

    folder_ts = str(time())
    output_dir = os.path.join(args.output_dir, folder_ts)
    output_masked_dir = os.path.join(args.output_dir, folder_ts, 'masked')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_masked_dir, exist_ok=True)

    if valid_detections == 0:
        print('No objects detected with sufficient confidence. Exiting.')
        return

    # Load the model
    print(f'Loading model from {args.model_path}...')
    model = keras.models.load_model(args.model_path)
    print('Model loaded successfully!')

    classifier = ImageClassifier(model, args.dataset)

    print('Drawing detections with bounding boxes...')
    detection_boxes = coco_detector.get_detections_boxes(tensor_image[0], results)

    draw_and_save_detection_boxes(image_name, tensor_image, detection_boxes, output_dir)

    # Crop detected objects
    print('Cropping detected objects...')
    cropped_images = coco_detector.get_detections(tensor_image[0], results)

    print('Proceed predicting...')
    predictions_result = proceed_predictions(cropped_images, classifier, output_dir)

    # Handle models with masks
    print('\nModel supports instance segmentation masks!')
    masks = results.masks
    predictions_result_masked = None

    # Draw masks
    if masks is not None:
        print('Drawing detections with masks...')
        detection_masks = coco_detector.get_mask_detections_boxes(tensor_image[0], results)

        draw_and_save_detection_boxes(image_name, tensor_image, detection_masks, output_masked_dir)

        # Crop with masks
        print('Cropping with masks...')
        cropped_mask_images = coco_detector.get_mask_detections(tensor_image[0], results)

        print('Proceed predicting with masks...')
        predictions_result_masked = proceed_predictions(cropped_mask_images, classifier, output_masked_dir)

    print('Object detection completed!')

    ollama_client = OllamaClient()
    get_info_for_prediction(predictions_result, predictions_result_masked, ollama_client)

    gem_client = GemClient()
    get_info_for_prediction(predictions_result, predictions_result_masked, gem_client)


if __name__ == '__main__':
    main()