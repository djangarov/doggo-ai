
"""
LLMs package.

Contains LLMs integrations modules
"""

__version__ = '1.0.0'

from .chat_client_interface import ChatClientInterface
from .ollama_client import OllamaClient
from .gem_client import GemClient
from .image_client_interface import ImageClientInterface
from .gem_image_client import GemImageClient
from .client_factory import ClientFactory

__all__ = [
    'ImageClientInterface',
    'ChatClientInterface',
    'OllamaClient',
    'GemClient',
    'GemImageClient',
    'ClientFactory',
]

def list_llms() -> list[str]:
    """List available LLM modules."""
    return [
        'ImageClientInterface',
        'ChatClientInterface',
        'OllamaClient',
        'GemClient',
        'GemImageClient',
        'ClientFactory',
    ]