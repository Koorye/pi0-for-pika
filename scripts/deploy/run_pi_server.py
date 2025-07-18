import argparse
import logging
import socket

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def create_policy(args):
    return _policy_config.create_trained_policy(
        _config.get_config(args.config), args.checkpoint, default_prompt=args.default_prompt
    )


def main(args):
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a WebSocket server for a trained policy.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file for the policy.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file for the policy.",
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="Default prompt for the policy (optional).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record policy interactions (default: False)",
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    main(args)
