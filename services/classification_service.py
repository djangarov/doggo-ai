import json
import os
import re
from time import time

from matplotlib import patches, pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from llms.chat_client_interface import ChatClientInterface
from llms.prompts import BREED_DETAILS_TASK
from processors.coco_object_detector import COCOObjectDetector
from processors.image_classifier import ImageClassifier


class ClassificationService:
    def __init__(self, detector: COCOObjectDetector, classifier: ImageClassifier) -> None:
        self.detector = detector
        self.classifier = classifier

    def detect_objects(self, image_path: str) -> tuple|None:
        """
        Detect objects in the given image using the detector.

        Args:
            image_path (str): The path to the image file.

        Returns:
            A tuple containing the tensor image, image name, and detected objects.
        """
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
        """
        Get detection boxes for the detected objects.

        Args:
            tensor_image (tf.Tensor): The input tensor image.
            detected_objects (dict): The detected objects.

        Returns:
            dict: The detection boxes for the detected objects.
        """
        return self.detector.get_detections_boxes(tensor_image, detected_objects)

    def get_mask_detections_boxes(self, tensor_image: tf.Tensor, detected_objects: dict) -> dict:
        """
        Get detection boxes from masks for the detected objects.

        Args:
            tensor_image (tf.Tensor): The input tensor image.
            detected_objects (dict): The detected objects.

        Returns:
            dict: The detection boxes for the detected objects.
        """
        return self.detector.get_mask_detections_boxes(tensor_image, detected_objects)

    def get_detections(self, tensor_image: tf.Tensor, detected_objects: dict) -> dict:
        """
        Get detections for the detected objects.

        Args:
            tensor_image (tf.Tensor): The input tensor image.
            detected_objects (dict): The detected objects.

        Returns:
            dict: The detections for the detected objects.
        """
        return self.detector.get_detections(tensor_image, detected_objects)

    def draw_and_save_detection_boxes(self, image_name: str, tensor_image: tf.Tensor, detection_boxes: dict, output_dir: str) -> None:
        """
        Draw detection boxes around the detected objects on the image and save the result.

        Args:
            image_name (str): The name of the image.
            tensor_image (tf.Tensor): The input tensor image.
            detection_boxes (dict): The detection boxes for the detected objects.
            output_dir (str): The directory to save the output image.

        Returns:
            None
        """
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

    def proceed_predictions(self, cropped_images: dict, classifier: ImageClassifier, output_dir: str) -> list[dict]:
        """
        Process cropped images and make predictions using the classifier. Store the results in JSON files.

        Args:
            cropped_images (dict): The cropped images to be classified.
            classifier (ImageClassifier): The image classifier.
            output_dir (str): The directory to save the output JSON files.

        Returns:
            list[dict]: A list of prediction results.
        """
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
            print(f'Image saved at: {filepath}')
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

    def get_top_prediction_class_name(self, predictions: list[dict], predictions_masked: list[dict] | None) -> set[str]:
        """
        Compare predictions from normal and masked models and return top class names.

        Args:
            predictions (list[dict]): Predictions from the normal model.
            predictions_masked (list[dict] | None): Predictions from the masked model.

        Returns:
            set[str]: A set of top predicted class names.
        """
        top_class_names = set()

        for idx, prediction in enumerate(predictions):
            top_prediction = prediction['predicted_class_name']

            if predictions_masked is not None:
                top_prediction = prediction['predicted_class_name'] if prediction['confidence'] >= predictions_masked[idx]['confidence'] else predictions_masked[idx]['predicted_class_name']

            top_class_name = re.sub(r'[-_]', ' ', re.sub(r'^[^-]*-', '', top_prediction))
            top_class_names.add(top_class_name)

        return top_class_names

    def get_info_for_prediction(self, predictions: set[str], llm_client: ChatClientInterface) -> None:
        """
        Get detailed information for each prediction using the provided LLM client.

        Args:
            predictions (set[str]): The set of predicted class names.
            llm_client (ChatClientInterface): The LLM client to use for fetching information.

        Returns:
            None
        """
        for prediction in predictions:
            print('\n' + '='*50)
            print(f'Asking LLM ({llm_client.__class__.__name__}) for details about: {prediction}')
            print('\n' + '='*50)
            question = BREED_DETAILS_TASK.format(breed=prediction)
            messages = llm_client.build_initial_session([question])
            llm_client.stream_chat(messages)