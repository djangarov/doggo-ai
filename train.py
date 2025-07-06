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
NUM_CATEGORIES = 6
TEST_SIZE = 0.4


def main():
    # Get image arrays and labels for all image files
    images, labels = load_data("./datasets")

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    filename = "dental_model.keras"
    model.save(filename)
    print(f"Model saved to {filename}.")


def load_data(data_dir: str) -> tuple:
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.

    0 - Unknown
    1 - Cavitya
    2 - Fillings
    3 - Impacted Tooth
    4 - Implant
    5 - Normal
    """
    images = []
    labels = []

    with os.scandir(data_dir) as dirs:
        for dir in dirs:
            if dir.is_dir():
                label = int(dir.name)

                with os.scandir(dir) as files:
                    for file in files:
                        image = cv2.imread(file.path)
                        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                        images.append(image)
                        labels.append(label)


    return (images, labels)


def get_model() -> Model:
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # First convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(32, 3, activation='relu')(img_input)
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
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    return model


if __name__ == "__main__":
    main()