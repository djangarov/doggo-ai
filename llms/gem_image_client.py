import os
from google import genai
from google.genai import types

from llms.image_client_interface import ImageClientInterface


MODEL = 'gemini-2.5-flash'

class GemImageClient(ImageClientInterface):
    def __init__(self, model: str = MODEL, api_key: str = None) -> None:
        """
        Initialize the GemImageClient with a model and API key.

        Args:
            model (str): The model to use for image generation. Defaults to MODEL.
            api_key (str|None): The API key for Gemini. If None, will use GEMINI_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Returns:
            None
        """
        self.model = model
        api_key = api_key or os.getenv('GEMINI_API_KEY')

        if not api_key:
            raise ValueError('GEMINI_API_KEY not found.')

        self.client = genai.Client(api_key=api_key)

    def generate(self, message: str, store_path: str|None = None) -> None:
        try:
            response: types.GenerateContentResponse = self.client.models.generate_content(
                model=self.model,
                contents=[message],
            )

            self._handle_image_generation(response, store_path)
        except Exception as e:
            print(f"An error occurred during gemini image generation: {e}")

    def _build_config(self) -> types.GenerateContentConfig:
        """
        Build configuration for Gemini image generation

        Args:
            None

        Returns:
            types.GenerateContentConfig: The configuration for Gemini image generation
        """
        return types.GenerateContentConfig(
            image_config=types.ImageConfig(
                aspect_ratio='16:9',
                image_size='1k',
            ),
        )

    def _handle_image_generation(self, response: types.GenerateContentResponse, store_path: str|None = None) -> None:
        """
        Handle image generation response from Gemini

        Args:
            response (types.GenerateContentResponse): The image generation response
            store_path (str|None): The path to store the generated image

        Returns:
            None
        """
        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = part.as_image()

                if store_path:
                    image.save(store_path + '.png')

                image.show()

