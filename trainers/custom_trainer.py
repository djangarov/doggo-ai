import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam

from trainers import BaseTrainer


class CustomTrainer(BaseTrainer):
    """
    Custom model trainer
    """

    def __init__(self):
        super().__init__(
            model_name="CustomModel",
            epochs=30,
            test_size=0.4,
            batch_size=32,
            image_width=150,
            image_height=150)  # Modelspecific settings

    def get_model(self, num_categories: int) -> Model:
        """
        Returns Custom transfer learning model
        """
        # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
        # the three color channels: R, G, and B
        img_input = layers.Input(shape=(self.img_width, self.img_height, 3))

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

        # Fourth convolution extracts 64 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Convolution2D(256, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Flatten feature map to a 1-dim tensor
        x = layers.GlobalAveragePooling2D()(x)

        # Create a fully connected layer with ReLU activation and 512 hidden units
        x = layers.Dense(512, activation='relu')(x)

        # Add a dropout rate of 0.5
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Create output layer with a single node and softmax activation
        output = layers.Dense(num_categories, activation='softmax')(x)

        # Configure and compile the model
        model = Model(img_input, output)

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model