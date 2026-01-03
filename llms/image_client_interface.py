class ImageClientInterface():
    def generate(self, message: str, store_path: str|None = None) -> None:
        """Send a message to the LLM to generate an image."""
        raise NotImplementedError("generate method not implemented.")