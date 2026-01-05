class ImageClientInterface():
    def generate(self, message: str, store_path: str|None = None) -> None:
        """
        Send a message to the LLM to generate an image.

        Args:
            message (str): The prompt for image generation
            store_path (str|None): The path to store the generated image

        Returns:
            None
        """
        raise NotImplementedError("generate method not implemented.")