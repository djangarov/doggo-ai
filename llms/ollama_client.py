from ollama import ChatResponse, chat

from llms import ClientInterface


MODEL = 'ministral-3'

class OllamaClient(ClientInterface):
    def __init__(self, model: str = MODEL) -> None:
        self.model = model

    def chat(self, message: str) -> None:
        response = chat(
            model=self.model,
            messages=self._build_messages(message),
            stream=False,
        )

        print(response.message.content)

    def stream_chat(self, message: str) -> None:
        stream = chat(
            model=self.model,
            messages=self._build_messages(message),
            stream=True,
        )

        self._handle_stream(stream)

    def _build_messages(self, message: str) -> list[dict]:
        return [{'role': 'user', 'content': message}]

    def _handle_stream(self, stream: ChatResponse) -> None:
        for chunk in stream:
            if chunk.message.content:
                print(chunk.message.content, end='', flush=True)

