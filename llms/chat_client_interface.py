class ChatClientInterface():
    def stream_chat(self, messages: list[str]) -> None:
        """Send a message to the LLM and return stream response"""
        raise NotImplementedError("stream_chat method not implemented.")

    def chat(self, messages: list[str]) -> str:
        """Send a message to the LLM and return the response"""
        raise NotImplementedError("chat method not implemented.")