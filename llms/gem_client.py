import os
from google import genai
from google.genai import types

from llms import ClientInterface


MODEL = 'gemini-2.5-flash'

class GemClient(ClientInterface):
    def __init__(self, model: str = MODEL, api_key: str = None) -> None:
        self.model = model
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError('GEMINI_API_KEY not found. Set it in .env file or pass it directly.')
        self.client = genai.Client(api_key=api_key)

    def chat(self, message: str) -> None:
        response = self.client.models.generate_content(
            model=self.model,
            contents=message,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        print(response.text)

    def stream_chat(self, message: str) -> None:
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=[message]
        )

        self._handle_stream(stream)

    def _handle_stream(self, stream: types.GenerateContentResponse) -> None:
        for chunk in stream:
            print(chunk.text, end='', flush=True)

