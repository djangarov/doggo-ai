
"""
LLMs package.

Contains LLMs integrations modules
"""

__version__ = '1.0.0'

from .client_interface import ClientInterface
from .ollama_client import OllamaClient
from .gem_client import GemClient

__all__ = [
    'ClientInterface',
    'OllamaClient',
    'GemClient',
]

def list_llms() -> list[str]:
    """List available LLM modules."""
    return [
        'ClientInterface',
        'OllamaClient',
        'GemClient',
    ]