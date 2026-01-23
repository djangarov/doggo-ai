from ollama import ChatResponse, ListResponse, chat, list as tags

from llms.chat_client_interface import ChatClientInterface, ChatSessionInterface


MODEL = 'ministral-3'

class OllamaSession(ChatSessionInterface):
    def add_message(self, message: str) -> None:
        self.messages.append({'role': 'user', 'content': message})

class OllamaClient(ChatClientInterface):
    def __init__(self, personality: str, model: str = MODEL) -> None:
        """
        Initialize the OllamaClient with a personality and model.

        Args:
            personality (str): The personality for the chat session.
            model (str): The model to use for chat. Defaults to MODEL.

        Returns:
            None
        """
        self.model = model
        self.personality = personality

    def chat(self, session: OllamaSession) -> str:
        try:
            response: ChatResponse = chat(
                model=self.model,
                messages=session.messages,
                stream=False,
                options={'temperature': 0.4},
            )

            return response.message.content
        except Exception as e:
            print(f"An error occurred during ollama chat: {e}")

    def stream_chat(self, session: OllamaSession) -> str:
        try:
            stream: ChatResponse = chat(
                model=self.model,
                messages=session.messages,
                stream=True,
                options={'temperature': 0.4},
            )

            return self._handle_stream(stream)
        except Exception as e:
            print(f"An error occurred during ollama stream chat: {e}")

    def _get_personality(self) -> dict:
        """
        Return the personality as a system message

        Args:
            None

        Returns:
            dict: The personality system message
        """
        return {'role': 'system', 'content': self.personality}

    def build_initial_session(self, messages: list[str]|None = None) -> OllamaSession:
        """
        Build messages for the LLM. Add personallity and append user messages if any.

        Args:
            messages (list[str]|None): Initial messages for the session

        Returns:
            OllamaSession: The chat session
        """
        initial = [self._get_personality()]

        if messages:
            initial.extend([{'role': 'user', 'content': message} for message in messages])

        return OllamaSession(initial)

    def get_available_models(self) -> list[str]:
        """
        Get available tags from the Ollama model.

        Args:
            None

        returns:
            list[str]: List of available tags
        """
        try:
            response: ListResponse = tags()

            return [model.model for model in response.models]
        except Exception as e:
            print(f"An error occurred while fetching available models")
            raise e

    def _handle_stream(self, stream: ChatResponse) -> str:
        """
        Handle streaming response from Ollama

        Args:
            stream (ChatResponse): The streaming response

        Returns:
            str: The complete response text
        """
        response = ''

        for chunk in stream:
            if chunk.message.content:
                print(chunk.message.content, end='', flush=True)
                response += chunk.message.content

        return response


