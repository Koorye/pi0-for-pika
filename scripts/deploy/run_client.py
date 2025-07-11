import argparse

from src.deploy.client import Client
from src.deploy.piper import Piper


def main(args):
    piper = Piper(args.can)
    client = Client(
        piper, 
        host=args.host, 
        port=args.port, 
        frequency=args.frequency
    )
    client.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy the OpenPI client.')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address of the server.')
    parser.add_argument('--port', type=int, default=8000, help='Port number of the server.')
    parser.add_argument('--frequency', type=int, default=10, help='Frequency of actions in Hz.')
    parser.add_argument('--can', type=str, default='can0', help='CAN interface to use.')
    args = parser.parse_args()
    main(args)
    