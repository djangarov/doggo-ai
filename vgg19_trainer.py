import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from base_trainer import BaseTrainer


class VGG19Trainer(BaseTrainer):
    """
    VGG19 model trainer
    """

    def __init__(self):
        super().__init__(
            model_name="VGG19",
            epochs=50,
            test_size=0.4,
            batch_size=32,
            # batch_size=12, # Reduced due to larger image size
            image_width=224,
            image_height=224)  # VGG19 specific settings

    def get_model(self, num_categories: int) -> Model:
        """
        Returns VGG19 transfer learning model
        """
        # VGG19 base model
        base_model = tf.keras.applications.VGG19(
            input_shape=(self.img_width, self.img_height, 3),
            include_top=False,
            weights='imagenet'
        )

        base_model.trainable = False  # Freeze base model

        # Add preprocessing and data augmentation
        inputs = tf.keras.Input(shape=(self.img_width, self.img_height, 3))

        # Data augmentation
        x = tf.keras.layers.RandomFlip("horizontal")(inputs)
        x = tf.keras.layers.RandomRotation(0.15)(x)
        x = tf.keras.layers.RandomZoom(0.15)(x)
        x = tf.keras.layers.RandomBrightness(0.1)(x)

        # VGG19 preprocessing
        x = tf.keras.applications.vgg19.preprocess_input(x)

        # Base model
        x = base_model(x, training=False)

        # Classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
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