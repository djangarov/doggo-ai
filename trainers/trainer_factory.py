from trainers.vgg19_trainer import VGG19Trainer
from trainers.resnet50_trainer import ResNet50Trainer
from trainers.inception_v3_trainer import InceptionV3Trainer
from trainers.custom_trainer import CustomTrainer
from trainers.inception_resnet_v2_trainer import InceptionResNetV2Trainer
from trainers.efficientnet_b7_trainer import EfficientNetB7Trainer


class TrainerFactory:
    """
    Factory class for creating model trainers
    """

    @staticmethod
    def create_trainer(model_type: str):
        """
        Create trainer instance based on model type
        """
        trainers = {
            'vgg19': VGG19Trainer,
            'resnet50': ResNet50Trainer,
            'inception_v3': InceptionV3Trainer,
            'custom': CustomTrainer,
            'inception_resnet_v2': InceptionResNetV2Trainer,
            'efficientnet_b7': EfficientNetB7Trainer
        }

        if model_type not in trainers:
            raise ValueError(f'Unknown model type: {model_type}')

        return trainers[model_type]()

    @staticmethod
    def get_available_models():
        """
        Get list of available model types
        """
        return ['vgg19', 'resnet50', 'inception_v3', 'custom', 'inception_resnet_v2', 'efficientnet_b7']