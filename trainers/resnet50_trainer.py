import keras

from trainers import BaseTrainer


class ResNet50Trainer(BaseTrainer):
    """
    ResNet50 model trainer
    """

    def __init__(self) -> None:
        super().__init__(
            model_type='ResNet50',
            epochs=20,
            batch_size=16,  # Adjusted for ResNet50 memory requirements
            image_width=224,
            image_height=224)  # ResNet50 specific settings

    def get_model(self, num_categories: int) -> keras.Model:
        """
        Returns ResNet50 transfer learning model with data augmentation.

        Args:
            num_categories: Number of output classes for classification

        Returns:
            Compiled Keras model with frozen ResNet50 base and custom head
        """
        input_shape = (self.img_height, self.img_width, 3)

        # ResNet50 base model pre-trained on ImageNet
        base_model = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,  # Exclude top classification layer
            weights='imagenet'  # Use ImageNet pre-trained weights
        )

        base_model.trainable = False  # Freeze base model for transfer learning

        inputs = keras.Input(shape=input_shape)

        # Data augmentation layers
        x = keras.layers.RandomFlip('horizontal')(inputs)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
        x = keras.layers.RandomContrast(0.1)(x)
        x = keras.layers.RandomBrightness(0.1)(x)
        x = keras.layers.RandomTranslation(0.1, 0.1)(x)

        # ResNet50 preprocessing and feature extraction
        x = keras.applications.resnet50.preprocess_input(x)
        x = base_model(x, training=False)

        # Add custom convolutional layer for Grad-CAM visualization
        x = keras.layers.Conv2D(
            512, (1, 1),
            activation='relu',
            name='grad_cam_conv'
        )(x)

        # Classification head with regularization
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        # Add L2 regularization
        x = keras.layers.Dense(512,
                               activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)

        # Output layer for multi-class classification
        outputs = keras.layers.Dense(num_categories,
                                     activation='softmax',
                                     kernel_regularizer=keras.regularizers.l2(0.001))(x)

        model = keras.Model(inputs, outputs)

        # Compile model with lower learning rate for transfer learning
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.0003,
                weight_decay=0.0002
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model
