import sys
sys.path.append('.')

import argparse

from src.deploy.dummy_client import DummyClient


def main(args):
    client = DummyClient(
        host=args.host, 
        port=args.port, 
        frequency=args.frequency
    )
    client.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy the OpenPI client.')
    parser.add_argument(
        '--host', 
        type=str, 
        default='127.0.0.1', 
        help='Host address of the server.'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000, 
        help='Port number of the server.'
    )
    parser.add_argument(
        '--frequency', 
        type=int, 
        default=10, 
        help='Frequency of actions in Hz.'
    )
    args = parser.parse_args()
    main(args)
