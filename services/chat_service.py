from llms.chat_client_interface import ChatClientInterface


class ChatService:
    def __init__(self, chat_client: ChatClientInterface) -> None:
        self.client = chat_client

    def start_conversation(self) -> None:
        session_messages = self.client.build_initial_session()

        while True:
            user_input = input('You: ')

            if user_input.lower() == 'exit':
                print('Exiting chat...')
                break

            session_messages.add_message(user_input)

            print('Assistant: ', end='', flush=True)
            response = self.client.stream_chat(session_messages)
            session_messages.add_message(response)
            print()  # For newline after streaming response