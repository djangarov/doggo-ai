class ChatSessionInterface():
    def __init__(self, messages: dict) -> None:
        self.messages = messages

    def add_message(self, message: str) -> None:
        """Add a message to the chat session"""
        raise NotImplementedError("add_message method not implemented.")


class ChatClientInterface():
    def stream_chat(self, session: ChatSessionInterface) -> str:
        """Send a message to the LLM and return stream response"""
        raise NotImplementedError("stream_chat method not implemented.")

    def chat(self, session: ChatSessionInterface) -> str:
        """Send a message to the LLM and return the response"""
        raise NotImplementedError("chat method not implemented.")

    def build_initial_session(self, messages: list[str]|None = None) -> ChatSessionInterface:
        """Build session for the LLM"""
        raise NotImplementedError("build_initial_session method not implemented.")