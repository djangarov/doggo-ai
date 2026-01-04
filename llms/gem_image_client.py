import os
from google import genai
from google.genai import types

from llms.image_client_interface import ImageClientInterface


MODEL = 'gemini-2.5-flash'

class GemImageClient(ImageClientInterface):
    def __init__(self, model: str = MODEL, api_key: str = None) -> None:
        self.model = model
        api_key = api_key or os.getenv('GEMINI_API_KEY')

        if not api_key:
            raise ValueError('GEMINI_API_KEY not found.')

        self.client = genai.Client(api_key=api_key)

    def generate(self, message: str, store_path: str|None = None) -> None:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[message],
            )

            self._handle_image_generation(response, store_path)
        except Exception as e:
            print(f"An error occurred during gemini image generation: {e}")

    def _build_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            image_config=types.ImageConfig(
                aspect_ratio='16:9',
                image_size='1k',
            ),
        )

    def _handle_image_generation(self, response: types.GenerateContentResponse, store_path: str|None = None) -> None:
        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = part.as_image()

                if store_path:
                    image.save(store_path + '.png')

                image.show()

