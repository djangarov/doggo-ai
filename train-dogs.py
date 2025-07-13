import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

EPOCHS = 20
IMG_WIDTH = 150
IMG_HEIGHT = 150
NUM_CATEGORIES = 157
TEST_SIZE = 0.4
BATCH_SIZE = 32


def main():
    data_dir = "./datasets-dogs"

    find_problematic_files(data_dir)
    # Load image data from directory
    x_train, y_test = load_data(data_dir)

    # Get a compiled neural network
    model = get_model()
    model.summary()
    # # Fit model on training data
    model.fit(x_train, validation_data=y_test, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(y_test, verbose=2)

    # # Save model to file
    filename = "dogs_model.keras"
    model.save(filename)
    print(f"Model saved to {filename}.")

def validate_image_format(image_path):
    """
    Validate if an image can be decoded by TensorFlow
    """
    try:
        # Read the image file
        image_raw = tf.io.read_file(image_path)

        # Try to decode the image
        image = tf.image.decode_image(image_raw, channels=3)

        # Check if image has valid dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            return False

        return True
    except Exception as e:
        print(f"Invalid image {image_path}: {e}")
        return False

def find_problematic_files(data_dir):
    """
    Find files that might cause issues including broken images
    """
    problematic_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            # Check for files without extensions or with unusual extensions
            if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                problematic_files.append(file_path)
                print(f"Unsupported format: {file_path}")
                os.remove(file_path)
                continue

            # Validate image format using TensorFlow
            if not validate_image_format(file_path):
                problematic_files.append(file_path)
                print(f"Broken image found and removed: {file_path}")
                os.remove(file_path)

    return problematic_files

def load_data(data_dir: str) -> tuple:
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.
    """
    try:
        x_train = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE)

        y_test = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE)

        # Optimize dataset for performance
        x_train = x_train.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        y_test = y_test.prefetch(buffer_size=tf.data.AUTOTUNE)

        return x_train, y_test
    except tf.errors.InvalidArgumentError as e:
        print(f"Image format error: {e}")


def get_model() -> Model:
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Data augmentation using the following Keras preprocessing layers
    x = layers.RandomFlip('horizontal')(img_input)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # Standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling
    x = layers.Rescaling(1./255)(x)

    # First convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Convolution2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Flatten feature map to a 1-dim tensor
    x = layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)

    # Create output layer with a single node and softmax activation
    output = layers.Dense(NUM_CATEGORIES, activation='softmax')(x)

    # Configure and compile the model
    model = Model(img_input, output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    main()