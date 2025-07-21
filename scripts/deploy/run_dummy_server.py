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
        mode='+' + args.mode if not args.invert_direction else '-' + args.mode,
        delta=args.delta,
        use_multi_arm=args.use_multi_arm,
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
        "--mode",
        type=str,
        default='x',
        choices=['x', 'y', 'z', 'rx', 'ry', 'rz'],
        help="Mode of operation for the dummy server (default: '+x')",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=1000,
        help="Delta value for the action (default: 1000)",
    )
    parser.add_argument(
        "--invert-direction",
        action="store_true",
        help="Invert the direction of the action (default: False)",
    )
    parser.add_argument(
        "--use-multi-arm",
        action="store_true",
        help="Use the multi-arm dummy server instead of the standard one",
    )
    args = parser.parse_args()
    main(args)
