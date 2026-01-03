class ClientInterface():
    def stream_chat(self, message: str) -> None:
        """Send a message to the LLM and return stream response"""
        raise NotImplementedError("stream_chat method not implemented.")

    def chat(self, message: str) -> str:
        """Send a message to the LLM and return the response"""
        raise NotImplementedError("chat method not implemented.")