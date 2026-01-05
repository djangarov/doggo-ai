import os
from google import genai
from google.genai import types

from llms.chat_client_interface import ChatClientInterface, ChatSessionInterface

MODEL = 'gemini-2.5-flash'

class GemSession(ChatSessionInterface):
    def add_message(self, message: str) -> None:
        self.messages.append(types.Content(role='user', parts=[types.Part(text=message)]))

class GemClient(ChatClientInterface):
    def __init__(self, personality: str, model: str = MODEL, api_key: str = None) -> None:
        """
        Initialize the GemClient with a personality, model, and API key.

        Args:
            personality (str): The personality for the chat session.
            model (str): The model to use for chat. Defaults to MODEL.
            api_key (str|None): The API key for Gemini. If None, will use GEMINI_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Returns:
            None
        """
        self.model = model
        self.personality = personality
        api_key = api_key or os.getenv('GEMINI_API_KEY')

        if not api_key:
            raise ValueError('GEMINI_API_KEY not found.')

        self.client = genai.Client(api_key=api_key)

    def chat(self, session: GemSession) -> str:
        try:
            response: types.GenerateContentResponse = self.client.models.generate_content(
                model=self.model,
                contents=session.messages,
                config=self._build_config(),
            )

            return response.text
        except Exception as e:
            print(f"An error occurred during gemini chat: {e}")

    def stream_chat(self, session: GemSession) -> str:
        try:
            stream: types.GenerateContentResponse = self.client.models.generate_content_stream(
                model=self.model,
                contents=session.messages,
                config=self._build_config(),
            )

            self._handle_stream(stream)
        except Exception as e:
            print(f"An error occurred during gemini stream chat: {e}")

    def build_initial_session(self, messages: list[str]|None = None) -> GemSession:
        """
        Build session for the LLM

        Args:
            messages (list[str]|None): Initial messages for the session

        Returns:
            GemSession: The chat session
        """
        if messages is None:
            return GemSession([])

        return GemSession([
            types.Content(role='user', parts=[types.Part(text=message)])
            for message in messages
        ])

    def _build_config(self) -> types.GenerateContentConfig:
        """
        Build configuration for Gemini requests

        Args:
            None

        Returns:
            types.GenerateContentConfig: The configuration for Gemini requests
        """
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            system_instruction=self.personality,
            temperature=0.4,
        )

    def _handle_stream(self, stream: types.GenerateContentResponse) -> str:
        """
        Handle streaming response from Gemini

        Args:
            stream (types.GenerateContentResponse): The streaming response

        Returns:
            str: The complete response text
        """
        response = ''

        for chunk in stream:
            print(chunk.text, end='', flush=True)
            response += chunk.text

        return response