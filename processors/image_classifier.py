import os
import numpy as np
import tensorflow as tf
import keras

from processors import ImageProcessor


class ClassifierResult:
    def __init__(self, predicted_class: int, confidence: float, predictions: np.ndarray, class_names: list = None):
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.predictions = predictions
        self.class_names = class_names

    def get_top(self, k: int) -> list[tuple[int, float]]:
        """Get top k predictions."""
        top_indices = np.argsort(self.predictions)[-k:][::-1]

        return [(idx, self.predictions[idx]) for idx in top_indices]


class ImageClassifier(ImageProcessor):
    """Classifies images using a trained machine learning model."""

    def __init__(self, model: keras.Model, dataset_dir: str | None = None):
        self.model = model
        self.dataset_dir = dataset_dir

    def __load_class_names(self) -> list | None:
        if self.dataset_dir is not None and not os.path.exists(self.dataset_dir):
            return None

        class_names = []
        for class_dir in sorted(os.listdir(self.dataset_dir)):
            if os.path.isdir(os.path.join(self.dataset_dir, class_dir)):
                class_names.append(class_dir)

        return class_names

    def predict_image(self, image: tf.Tensor) -> ClassifierResult | None:
        """
        Predict the class of a single image
        """
        try:
            # Get class names if dataset directory is provided
            class_names = self.__load_class_names()

            # Make prediction
            predictions = self.model.predict(image, verbose=0)

            # Get the predicted class
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            return ClassifierResult(predicted_class, confidence, predictions[0], class_names)

        except Exception as e:
            print(f'Error during prediction: {str(e)}')

            return None

