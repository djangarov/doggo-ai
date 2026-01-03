import os
from google import genai
from google.genai import types

from llms.chat_client_interface import ChatClientInterface
from llms.prompts import CONFIG_PERSONALITY_FIRST_TIME_DOG_OWNER


MODEL = 'gemini-2.5-flash'

class GemClient(ChatClientInterface):
    def __init__(self, model: str = MODEL, api_key: str = None) -> None:
        self.model = model
        api_key = api_key or os.getenv('GEMINI_API_KEY')

        if not api_key:
            raise ValueError('GEMINI_API_KEY not found.')

        self.client = genai.Client(api_key=api_key)

    def chat(self, message: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=message,
                config=self._build_config(),
            )

            return response.text
        except Exception as e:
            print(f"An error occurred during gemini chat: {e}")

    def stream_chat(self, message: str) -> None:
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=[message],
                config=self._build_config(),
            )

            self._handle_stream(stream)
        except Exception as e:
            print(f"An error occurred during gemini stream chat: {e}")

    def _build_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            system_instruction=CONFIG_PERSONALITY_FIRST_TIME_DOG_OWNER,
            temperature=0.4,
        )

    def _handle_stream(self, stream: types.GenerateContentResponse) -> None:
        for chunk in stream:
            print(chunk.text, end='', flush=True)
