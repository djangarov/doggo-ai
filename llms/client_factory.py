from llms.gem_client import GemClient
from llms.ollama_client import OllamaClient


class ClientFactory:
    """
    Factory class for LLM
    """

    @staticmethod
    def create_client(client_type: str, personality: str):
        """
        Create LLM client instance based on client type
        """
        llms = {
            'ollama': OllamaClient,
            'gemini': GemClient,
        }

        if client_type not in llms:
            raise ValueError(f'Unknown client type: {client_type}')

        return llms[client_type](personality)

    @staticmethod
    def get_available_clients():
        """
        Get list of available LLMs types
        """
        return ['ollama', 'gemini']