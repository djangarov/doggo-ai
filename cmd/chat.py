import argparse
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llms import ClientFactory, ChatClientInterface
from llms.prompts import CONFIG_PERSONALITY_PROFESSIONAL_DOG_TRAINER

load_dotenv()

def start_conversation(chat_client: ChatClientInterface) -> None:
    session_messages = chat_client.build_initial_session()

    while True:
        user_input = input('You: ')

        if user_input.lower() == 'exit':
            print('Exiting chat...')
            break

        session_messages.add_message(user_input)

        print('Assistant: ', end='', flush=True)
        response = chat_client.stream_chat(session_messages)
        session_messages.add_message(response)
        print()  # For newline after streaming response

def main():
    parser = argparse.ArgumentParser(description='Chat with AI assistant.')
    parser.add_argument('client_type', choices=ClientFactory.get_available_clients(), help='Type of client to use')

    args = parser.parse_args()

    print(f'Chatting using {args.client_type} client...')

    chat_client = ClientFactory.create_client(args.client_type, CONFIG_PERSONALITY_PROFESSIONAL_DOG_TRAINER)

    start_conversation(chat_client)

if __name__ == '__main__':
    main()