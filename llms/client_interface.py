class ClientInterface():
    def stream_chat(self, message: str) -> str:
        """Send a message to the LLM and return stream response"""
        pass

    def chat(self, message: str) -> str:
        """Send a message to the LLM and return the response"""
        pass