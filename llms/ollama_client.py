from ollama import ChatResponse, chat

from llms.chat_client_interface import ChatClientInterface


MODEL = 'ministral-3'

class OllamaClient(ChatClientInterface):
    def __init__(self, personality: str, model: str = MODEL) -> None:
        self.model = model
        self.personality = personality

    def chat(self, messages: list[str]) -> str:
        try:
            response = chat(
                model=self.model,
                messages=self._build_messages(messages),
                stream=False,
                options={'temperature': 0.4},
            )

            return response.message.content
        except Exception as e:
            print(f"An error occurred during ollama chat: {e}")

    def stream_chat(self, messages: list[str]) -> str:
        try:
            stream = chat(
                model=self.model,
                messages=self._build_messages(messages),
                stream=True,
                options={'temperature': 0.4},
            )

            return self._handle_stream(stream)
        except Exception as e:
            print(f"An error occurred during ollama stream chat: {e}")

    def _get_personality(self) -> dict:
        return {'role': 'system', 'content': self.personality}

    def _build_messages(self, messages: list[str]) -> list[dict]:
        return [self._get_personality()] + [{'role': 'user', 'content': message} for message in messages]

    def _handle_stream(self, stream: ChatResponse) -> str:
        response = ''

        for chunk in stream:
            if chunk.message.content:
                print(chunk.message.content, end='', flush=True)
                response += chunk.message.content

        return response


