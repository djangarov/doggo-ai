import keras

from trainers import AbstractTrainer


class CustomTrainer(AbstractTrainer):
    """
    Custom model trainer
    """

    def __init__(self) -> None:
        super().__init__(
            model_type='CustomModel',
            epochs=30,
            batch_size=32,
            image_width=150,
            image_height=150) # Model specific settings

    def get_model(self, num_categories: int) -> keras.Model:
        """
        Returns custom CNN model with data augmentation and dropout regularization.

        Args:
            num_categories: Number of output classes for classification

        Returns:
            Compiled Keras model ready for training
        """
        # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
        # the three color channels: R, G, and B
        img_input = keras.layers.Input(shape=(self.img_height, self.img_width, 3))

        # Data augmentation using the following Keras preprocessing layers
        x = keras.layers.RandomFlip('horizontal')(img_input)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)

        # Standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling
        x = keras.layers.Rescaling(1./255)(x)

        # First convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

        # Second convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

        # Third convolution extracts 128 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = keras.layers.Conv2D(128, 3, activation='relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

        # Fourth convolution extracts 256 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = keras.layers.Conv2D(256, 3, activation='relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

        # Flatten feature map to a 1-dim tensor
        x = keras.layers.GlobalAveragePooling2D()(x)

        # Create a fully connected layer with ReLU activation and 512 hidden units
        x = keras.layers.Dense(512, activation='relu')(x)

        # Add a dropout rate of 0.5
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)

        # Create output layer with a single node and softmax activation
        output = keras.layers.Dense(num_categories, activation='softmax')(x)

        # Configure and compile the model
        model = keras.Model(img_input, output)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model