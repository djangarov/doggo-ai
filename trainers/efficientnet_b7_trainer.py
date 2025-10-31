import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam

from trainers import BaseTrainer


class EfficientNetB7Trainer(BaseTrainer):
    """
    EfficientNetB7 model trainer
    """

    def __init__(self):
        super().__init__(
            model_name="EfficientNetB7",
            epochs=50,
            test_size=0.4,
            batch_size=10,  # Adjusted for EfficientNetB7 memory requirements
            # batch_size=4,
            image_width=600,
            image_height=600)  # EfficientNetB7 specific settings

    def get_model(self, num_categories: int) -> Model:
        """
        Returns EfficientNetB7 transfer learning model
        """
        # EfficientNetB7 base model
        base_model = keras.applications.EfficientNetB7(
            input_shape=(self.img_width, self.img_height, 3),
            include_top=False,
            weights='imagenet'
        )

        base_model.trainable = False  # Freeze base model

        inputs = keras.Input(shape=(self.img_width, self.img_height, 3))

        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomBrightness(0.1)(x)

        # EfficientNet preprocessing
        x = keras.applications.efficientnet.preprocess_input(x)
        x = base_model(x, training=False)

        # Add a named convolutional layer for Grad-CAM access
        x = layers.Conv2D(
            512, (1, 1),
            activation='relu',
            name='grad_cam_conv'
        )(x)

        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        outputs = keras.layers.Dense(num_categories, activation='softmax')(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model