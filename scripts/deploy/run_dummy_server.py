import sys
sys.path.append('.')

import argparse
import logging
import socket

from src.deploy.servers import WebsocketDummyServer


def main(args):
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketDummyServer(
        host="0.0.0.0",
        port=args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a dummy server for testing purposes.")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--use-multi-arm",
        action="store_true",
        help="Use the multi-arm dummy server instead of the standard one",
    )
    args = parser.parse_args()
    main(args)
