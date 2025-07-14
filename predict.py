import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import argparse

# Constants (should match your training script)
IMG_WIDTH = 150
IMG_HEIGHT = 150
NUM_CATEGORIES = 157

def get_class_names(data_dir) -> list | None:
    """
    Get class names from dataset directory structure
    """
    if not os.path.exists(data_dir):
        return None

    class_names = []
    for class_dir in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, class_dir)):
            class_names.append(class_dir)
    return class_names

def preprocess_image(image_path) -> tf.Tensor:
    """
    Preprocess a single image for prediction - MUST match training preprocessing
    """
    # Read the image file
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return img_array

def predict_image(model_path, image_path, dataset_dir=None):
    """
    Predict the class of a single image
    """
    try:
        # Load the model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        print("Model loaded successfully!")

        # Print model summary to check if it's properly trained
        print("\nModel Summary:")
        model.summary()

        # Check model weights (should not be all zeros)
        first_layer_weights = model.layers[0].get_weights()
        if len(first_layer_weights) > 0:
            print(f"First layer weight range: {first_layer_weights[0].min():.6f} to {first_layer_weights[0].max():.6f}")

        # Get class names if dataset directory is provided
        class_names = None
        if dataset_dir:
            class_names = get_class_names(dataset_dir)
            if class_names:
                print(f"Found {len(class_names)} classes")

        # Preprocess the image
        print(f"Processing image: {image_path}")
        processed_image = preprocess_image(image_path)

        # Make prediction
        print("Making prediction...")
        predictions = model.predict(processed_image, verbose=0)

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_confidences = predictions[0][top_5_indices]

        # Print results
        print("\n" + "="*50)
        print(f"PREDICTION RESULTS")
        print("="*50)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted class ID: {predicted_class}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

        if class_names and predicted_class < len(class_names):
            print(f"Predicted class name: {class_names[predicted_class]}")
            print("\nTop 5 predictions:")
            for i, (idx, conf) in enumerate(zip(top_5_indices, top_5_confidences)):
                if idx < len(class_names):
                    print(f"{i+1}. {class_names[idx]}: {conf:.4f} ({conf*100:.2f}%)")
        else:
            print("\nTop 5 predictions:")
            for i, (idx, conf) in enumerate(zip(top_5_indices, top_5_confidences)):
                print(f"{i+1}. Class {idx}: {conf:.4f} ({conf*100:.2f}%)")

        return predicted_class, confidence

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Predict image class using trained model')
    parser.add_argument('model_path', help='Path to the trained model file (.keras)')
    parser.add_argument('image_path', help='Path to the image to predict')
    parser.add_argument('dataset', help='Path to dataset directory for class names')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        sys.exit(1)

    if not os.path.exists(args.dataset):
        print(f"Warning: Dataset directory '{args.dataset}' not found! Class names will not be displayed.")

    # Make prediction
    predict_image(args.model_path, args.image_path, args.dataset)

if __name__ == "__main__":
    main()