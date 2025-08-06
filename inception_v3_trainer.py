import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from base_trainer import BaseTrainer


class InceptionV3Trainer(BaseTrainer):
    """
    InceptionV3 model trainer
    """

    def __init__(self):
        super().__init__(
            model_name="InceptionV3",
            epochs=50,
            test_size=0.4,
            batch_size=32,
            # batch_size=16, # Reduced due to larger image size
            image_width=299,
            image_height=299)  # InceptionV3 specific settings

    def get_model(self, num_categories: int) -> Model:
        """
        Returns InceptionV3 transfer learning model
        """
        # InceptionV3 base model
        base_model = tf.keras.applications.InceptionV3(
            input_shape=(self.img_width, self.img_height, 3),
            include_top=False,
            weights='imagenet'
        )

        base_model.trainable = False  # Freeze base model

        inputs = tf.keras.Input(shape=(self.img_width, self.img_height, 3))

        # Data augmentation
        x = tf.keras.layers.RandomFlip("horizontal")(inputs)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
        x = tf.keras.layers.RandomBrightness(0.1)(x)

        # InceptionV3 preprocessing
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        x = base_model(x, training=False)

        # Classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        outputs = tf.keras.layers.Dense(num_categories, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model