from ollama import ChatResponse, chat

from llms.chat_client_interface import ChatClientInterface
from llms.prompts import CONFIG_PERSONALITY_FIRST_TIME_DOG_OWNER


MODEL = 'ministral-3'

class OllamaClient(ChatClientInterface):
    def __init__(self, model: str = MODEL) -> None:
        self.model = model

    def chat(self, message: str) -> str:
        try:
            response = chat(
                model=self.model,
                messages=self._build_messages(message),
                stream=False,
                options={'temperature': 0.4},
            )

            return response.message.content
        except Exception as e:
            print(f"An error occurred during ollama chat: {e}")

    def stream_chat(self, message: str) -> None:
        try:
            stream = chat(
                model=self.model,
                messages=self._build_messages(message),
                stream=True,
                options={'temperature': 0.4},
            )

            self._handle_stream(stream)
        except Exception as e:
            print(f"An error occurred during ollama stream chat: {e}")

    def _build_messages(self, message: str) -> list[dict]:
        return [
            {'role': 'system', 'content': CONFIG_PERSONALITY_FIRST_TIME_DOG_OWNER},
            {'role': 'user', 'content': message}
        ]

    def _handle_stream(self, stream: ChatResponse) -> None:
        for chunk in stream:
            if chunk.message.content:
                print(chunk.message.content, end='', flush=True)

