import os
import sys
import argparse
import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processors import ImageClassifier


def main():
    parser = argparse.ArgumentParser(description='Predict image class using trained model')
    parser.add_argument('model_path', help='Path to the trained model file (.keras)')
    parser.add_argument('image_path', help='Path to the image to predict')
    parser.add_argument('dataset', help='Path to dataset directory for class names')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f'Error: Model file {args.model_path} not found!')
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f'Error: Image file {args.image_path} not found!')
        sys.exit(1)

    if not os.path.exists(args.dataset):
        print(f'Warning: Dataset directory {args.dataset} not found! Class names will not be displayed.')

    # Load the model
    print(f'Loading model from {args.model_path}...')
    model = keras.models.load_model(args.model_path)
    print('Model loaded successfully!')

    classifier = ImageClassifier(model, args.dataset)

    # Preprocess the image
    print(f'Processing image: {args.image_path}')
    processed_image = classifier.preprocess_image(args.image_path, classifier.model.input_shape[1:3])

    # Make prediction
    print('Making prediction...')
    prediction_result = classifier.predict_image(processed_image)

     # Get top 5 predictions
    top_5_predictions =  prediction_result.get_top(5)

    # Print results
    print('\n' + '='*50)
    print(f'PREDICTION RESULTS')
    print('='*50)
    print(f'Predicted class ID: {prediction_result.predicted_class}')
    print(f'Confidence: {prediction_result.confidence:.4f} ({prediction_result.confidence*100:.2f}%)')

    if prediction_result.class_names and prediction_result.predicted_class < len(prediction_result.class_names):
        print(f'Predicted class name: {prediction_result.class_names[prediction_result.predicted_class]}')
        print('\nTop 5 predictions:')
        for i, (idx, conf) in enumerate(top_5_predictions):
            if idx < len(prediction_result.class_names):
                print(f'{i+1}. {prediction_result.class_names[idx]}: {conf:.4f} ({conf*100:.2f}%)')
    else:
        print('\nTop 5 predictions:')
        for i, (idx, conf) in enumerate(top_5_predictions):
            print(f'{i+1}. Class {idx}: {conf:.4f} ({conf*100:.2f}%)')

if __name__ == '__main__':
    main()