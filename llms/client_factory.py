from llms.chat_client_interface import ChatClientInterface
from llms.gem_client import GemClient
from llms.ollama_client import OllamaClient


class ClientFactory:
    """
    Factory class for LLM
    """

    @staticmethod
    def create_client(client_type: str, personality: str) -> ChatClientInterface:
        """
        Create LLM client instance based on client type

        Args:
            client_type (str): The type of LLM client
            personality (str): The personality for the LLM client

        Returns:
            ChatClientInterface: The LLM client instance
        """
        llms = {
            'ollama': OllamaClient,
            'gemini': GemClient,
        }

        if client_type not in llms:
            raise ValueError(f'Unknown client type: {client_type}')

        return llms[client_type](personality)

    @staticmethod
    def get_available_clients() -> list[str]:
        """
        Get list of available LLMs types

        Returns:
            list[str]: The list of available LLM client types
        """
        return ['ollama', 'gemini']