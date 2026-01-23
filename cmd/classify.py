import os
import sys
import argparse
from time import time
from dotenv import load_dotenv

import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.classification_service import ClassificationService
from llms import OllamaClient, GemClient, GemImageClient, ChatClientInterface, ImageClientInterface
from llms.prompts import BREED_DETAILS_TASK, CONFIG_PERSONALITY_FIRST_TIME_DOG_OWNER, DOG_TRAINER_WITH_SPECIFIC_BREAD_TASK
from processors import COCOObjectDetector, ImageClassifier

load_dotenv()

MIN_SCORE_THRESH = 0.3
TARGET_CLASS = 18  # Dog class ID


def generate_dog_trainer_image(predictions: set[str], llm_client: ImageClientInterface, output_dir: str) -> None:
    """Generate dog trainer images for each prediction using the provided image LLM client."""
    for prediction in predictions:
        print('\n' + '='*50)
        print(f'Asking LLM ({llm_client.__class__.__name__}) to generate image for breed: {prediction}')
        print('\n' + '='*50)
        prompt = DOG_TRAINER_WITH_SPECIFIC_BREAD_TASK.format(breed=prediction)
        llm_client.generate(prompt, output_dir + f'/{prediction.replace(' ', '_')}_trainer_image')

        return  # Generate only one image for the first prediction

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

    folder_ts = str(time())
    output_dir = os.path.join(args.output_dir, folder_ts)
    output_masked_dir = os.path.join(args.output_dir, folder_ts, 'masked')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_masked_dir, exist_ok=True)

    print('Loading model...')
    coco_detector = COCOObjectDetector(MIN_SCORE_THRESH, TARGET_CLASS)
    print('Model loaded!')

    # Load the model
    print(f'Loading model from {args.model_path}...')
    model = keras.models.load_model(args.model_path)
    print('Model loaded successfully!')

    chat_client = OllamaClient(CONFIG_PERSONALITY_FIRST_TIME_DOG_OWNER)

    try:
        chat_client.get_available_models()
    except Exception:
        chat_client = GemClient(CONFIG_PERSONALITY_FIRST_TIME_DOG_OWNER)

    classifier = ImageClassifier(model, args.dataset)

    classification_service = ClassificationService(coco_detector, classifier, chat_client)
    classification_service.clasify_image(args.image_path, output_dir, output_masked_dir)

    # gem_image_client = GemImageClient()
    # generate_dog_trainer_image(top_predictions, gem_image_client, output_dir)

if __name__ == '__main__':
    main()