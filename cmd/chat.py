import argparse
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llms.client_factory import ClientFactory
from llms.prompts import CONFIG_PERSONALITY_PROFESSIONAL_DOG_TRAINER
from services.chat_service import ChatService


def main():
    parser = argparse.ArgumentParser(description='Chat with AI assistant.')
    parser.add_argument('client_type', choices=ClientFactory.get_available_clients(), help='Type of client to use')

    args = parser.parse_args()

    print(f'Chatting using {args.client_type} client...')

    chat_client = ClientFactory.create_client(args.client_type, CONFIG_PERSONALITY_PROFESSIONAL_DOG_TRAINER)
    chat_service = ChatService(chat_client)
    chat_service.start_conversation()

if __name__ == '__main__':
    load_dotenv()

    main()