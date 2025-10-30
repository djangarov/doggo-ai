import os
import sys
import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

from PIL import Image, ImageDraw, ImageFont, ImageFile


def get_class_names(data_dir: str) -> list | None:
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

def preprocess_image(image_path: str, img_size: tuple, model: Model) -> tf.Tensor:
    """
    Preprocess a single image for prediction
    """
    # Read the image file
    img = tf.keras.utils.load_img(
        image_path, target_size=img_size
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return img_array

def draw_bounding_box_with_label(image: ImageFile, bbox: tuple, label: str, color=(0, 255, 0)):
    """
    Draw bounding box with label on image
    bbox: (x, y, width, height)
    """
    draw = ImageDraw.Draw(image)

    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h

    # Draw rectangle
    box_thickness = 3
    for i in range(box_thickness):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color, width=1)

    # Prepare label text
    label_text = f"{label}%"

    font = ImageFont.load_default(70)

    # Get text size
    bbox_text = draw.textbbox((0, 0), label_text, font=font, font_size=14)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]

    # Draw label background
    draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill=color)

    # Draw label text
    draw.text((x1+5, y1-text_height-2), label_text, fill=(255, 255, 255), font=font)

    return image

def get_object_coordinates_from_gradcam(model: Model, processed_image: tf.Tensor, original_image: ImageFile, threshold: float = 0.5):
    """
    Get object coordinates using the grad_cam_conv layer specifically
    """
    try:
        # Look for the grad_cam_conv layer specifically
        grad_cam_layer = None
        for layer in model.layers:
            if layer.name == 'grad_cam_conv':
                grad_cam_layer = layer
                break

        if grad_cam_layer is None:
            print("grad_cam_conv layer not found in model")
            return []

        print(f"Using layer: {grad_cam_layer.name}")

        # Fix: Use model.input instead of [model.inputs]
        grad_model = Model(
            inputs=model.input,  # Changed from [model.inputs]
            outputs=[grad_cam_layer.output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(processed_image)
            predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]

        # Get gradients and feature maps
        grads = tape.gradient(loss, conv_outputs)

        if grads is None or conv_outputs is None:
            print("Could not compute gradients")

            return []

        # Compute the weighted feature maps
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0)
        if tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))

        # Threshold and find contours
        binary_mask = (heatmap_resized > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coordinates = []
        min_area = (original_image.width * original_image.height) * 0.02  # Minimum 2% of image area

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter out very small regions
            if area > min_area:
                # Calculate confidence based on heatmap intensity in this region
                region_mask = np.zeros_like(heatmap_resized)
                cv2.rectangle(region_mask, (x, y), (x+w, y+h), 255, -1)
                region_confidence = np.mean(heatmap_resized[region_mask > 0])

                coordinates.append((x, y, w, h, region_confidence))

        # Sort by confidence (highest first)
        coordinates.sort(key=lambda x: x[4], reverse=True)

        return coordinates

    except Exception as e:
        print(f"Error getting object coordinates: {e}")

        return []

def create_enhanced_heatmap_overlay(model: Model, processed_image: tf.Tensor, original_image: ImageFile):
    """
    Create enhanced heatmap using the grad_cam_conv layer
    """
    try:
        # Look for the grad_cam_conv layer
        grad_cam_layer = None
        for layer in model.layers:
            if layer.name == 'grad_cam_conv':
                grad_cam_layer = layer
                break

        if grad_cam_layer is None:
            print("grad_cam_conv layer not found, falling back to default method")

            return None

        # Fix: Use model.input instead of [model.inputs]
        grad_model = Model(
            inputs=model.input,  # Changed from [model.inputs]
            outputs=[grad_cam_layer.output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(processed_image)
            predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return None

        # Compute heatmap
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Normalize
        heatmap = tf.maximum(heatmap, 0)
        if tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Resize to original image size
        heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))

        return heatmap_resized

    except Exception as e:
        print(f"Error creating enhanced heatmap: {e}")

        return None

def display_prediction_with_detection(image_path: str, predicted_class: int, confidence: float, class_names: list = None, model=None, processed_image=None):
    """
    Display the image with object detection bounding boxes
    """
    try:
        # Load original image
        img = Image.open(image_path)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        # 1. Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 2. Image with object detection bounding boxes
        img_with_boxes = img.copy()
        if model is not None and processed_image is not None:
            coordinates = get_object_coordinates_from_gradcam(model, processed_image, img, threshold=0.4)

            if coordinates:
                print(f"\nDetected {len(coordinates)} object regions:")
                for i, (x, y, w, h, region_conf) in enumerate(coordinates):
                    print(f"Region {i+1}: x={x}, y={y}, width={w}, height={h}, confidence={region_conf:.3f}")

                    # Get label
                    if class_names and predicted_class < len(class_names):
                        label = class_names[predicted_class]
                    else:
                        label = f"Class {predicted_class}"

                    # Choose color based on region confidence
                    if region_conf > 0.7:
                        color = (0, 255, 0)  # Green
                    elif region_conf > 0.5 and region_conf <= 0.7:
                        color = (255, 165, 0)  # Orange
                    else:
                        color = (255, 0, 0)  # Red

                    # Draw bounding box
                    img_with_boxes = draw_bounding_box_with_label(
                        img_with_boxes, (x, y, w, h), f"{label} Region {i+1}", color
                    )
            else:
                print("No significant object regions detected")

        axes[1].imshow(img_with_boxes)

        # Create title with prediction results
        if class_names and predicted_class < len(class_names):
            title = f"Detected: {class_names[predicted_class]}\nConfidence: {round(confidence*100, 2)}%"
        else:
            title = f"Detected: Class {predicted_class}\nConfidence: {round(confidence*100, 2)}%"

        axes[1].set_title(title, fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 3. Enhanced heatmap overlay
        if model is not None and processed_image is not None:
            heatmap = create_enhanced_heatmap_overlay(model, processed_image, img)
            if heatmap is not None:
                # Convert PIL to numpy for overlay
                img_array = np.array(img)

                # Create colored heatmap
                heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                # Blend images
                blended = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
                axes[2].imshow(blended)
                axes[2].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
            else:
                axes[2].imshow(img)
                axes[2].set_title('Heatmap Not Available', fontsize=14, fontweight='bold')
        else:
            axes[2].imshow(img)
            axes[2].set_title('Model Not Provided', fontsize=14, fontweight='bold')

        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error displaying image with detection: {str(e)}")

def predict_image(model_path: str, image_path: str, dataset_dir: str | None = None):
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

        # Get class names if dataset directory is provided
        class_names = None
        if dataset_dir:
            class_names = get_class_names(dataset_dir)
            if class_names:
                print(f"Found {len(class_names)} classes")

        # Preprocess the image
        print(f"Processing image: {image_path}")
        processed_image = preprocess_image(image_path, model.input_shape[1:3], model)

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

        # Display the image with object detection
        display_prediction_with_detection(image_path, predicted_class, confidence, class_names, model, processed_image)

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