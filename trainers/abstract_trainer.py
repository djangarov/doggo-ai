from abc import ABC, abstractmethod
import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


class AbstractTrainer(ABC):
    """
    Abstract base class for training different CNN models
    """

    def __init__(self,
                 model_type: str,
                 epochs: int = 50,
                 batch_size: int = 32,
                 image_width: int = 150,
                 image_height: int = 150) -> None:
        """
        Initialize the base trainer with the given parameters.
        """
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_width = image_width
        self.img_height = image_height

    @abstractmethod
    def get_model(self, num_categories: int) -> keras.Model:
        """Return the compiled model for training"""
        pass

    def load_data(self, data_dir: str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load image data from directory `data_dir`.
        """
        try:
            dataset = keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset='both',
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size)

            return dataset[0], dataset[1]
        except tf.errors.InvalidArgumentError as e:
            print(f'Image format error: {e}')
            raise

    def optimize_dataset(self,
                         train_dataset: tf.data.Dataset,
                         validation_dataset: tf.data.Dataset) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Optimize dataset for performance
        """
        train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset, validation_dataset

    def validate_image_format(self, image_path: str) -> bool:
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
            print(f'Invalid image {image_path}: {e}')
            return False

    def remove_problematic_files(self, data_dir: str) -> list[str]:
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
                    print(f'Unsupported format: {file_path}')
                    os.remove(file_path)
                    continue

                # Validate image format using TensorFlow
                if not self.validate_image_format(file_path):
                    problematic_files.append(file_path)
                    print(f'Broken image found and removed: {file_path}')
                    os.remove(file_path)

        return problematic_files

    def visualize_training(self,
                           model_name: str,
                           history: keras.callbacks.History) -> None:
        """
        Visualize training history
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        actual_epochs = len(acc)
        epochs_range = range(actual_epochs)

        save_path = f'{model_name}_training_plot.png'

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

        print(f'Training plot saved to {save_path}')

        plt.show(block=False)  # Make non-blocking
        plt.pause(10)  # Show for 10 seconds

    def save_model(self,
                   model: keras.Model,
                   model_name: str) -> None:
        """
        Save the trained model to a file
        """
        filename = f'{model_name}.keras'
        model.save(filename)
        print(f'Model saved to {filename}.')

    def get_callbacks(self, model_name: str) -> list[keras.callbacks.Callback]:
        """
        Get training callbacks
        """
        return [
            keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_accuracy',
                min_delta=0.001,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'{model_name}_best.keras',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]

    def train(self,
              dataset_dir: str,
              custom_model_name: str | None = None) -> tuple[keras.Model, keras.callbacks.History]:
        """
        Main training method
        """
        self.remove_problematic_files(dataset_dir)

        # Load image data from directory
        train_dataset, validation_dataset = self.load_data(dataset_dir)

        # Get actual number of categories from the dataset
        num_categories = len(train_dataset.class_names)
        print(f'Found {num_categories} categories in dataset')

        # Optimize dataset
        train_dataset, validation_dataset = self.optimize_dataset(train_dataset, validation_dataset)

        # Get a compiled neural network
        model = self.get_model(num_categories)
        model.summary()

        model_name = custom_model_name if custom_model_name else self.model_type
        callbacks = self.get_callbacks(model_name)

        # Fit model on training data
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate neural network performance
        model.evaluate(validation_dataset, verbose=2)
        self.visualize_training(model_name, history)

        # Save model to file
        self.save_model(model, model_name)

        print(f'{model_name} training completed!')

        return model, history