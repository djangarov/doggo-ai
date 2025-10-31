import argparse
import os

from trainers import ModelFactory


def main():
    parser = argparse.ArgumentParser(description='Train CNN models on images.')
    parser.add_argument('model_type', choices=ModelFactory.get_available_models(), help='Type of model to train')
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    parser.add_argument('--model_name', help='Model name to save as (optional)', default=None)

    args = parser.parse_args()

    if not os.path.exists(args.dataset_dir):
        print(f"Warning: Dataset directory '{args.dataset_dir}' not found! Class names will not be displayed.")

    trainer = ModelFactory.create_trainer(args.model_type)

    # Train the model
    model, history = trainer.train(args.dataset_dir, args.model_name)

    print(f"Training completed for {args.model_type}")


if __name__ == "__main__":
    main()