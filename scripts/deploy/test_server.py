import argparse
import numpy as np
import time

from openpi_client import websocket_client_policy


def main(args):
    example_observation = {
        'left_wrist_base_rgb': np.zeros((480, 640, 3), dtype=np.uint8),
        'right_wrist_base_rgb': np.zeros((480, 640, 3), dtype=np.uint8),
        'left_wrist_fisheye_rgb': np.zeros((480, 640, 3), dtype=np.uint8),
        'right_wrist_fisheye_rgb': np.zeros((480, 640, 3), dtype=np.uint8),
        'states': np.zeros(7, dtype=np.float32),
        'prompt': 'do something',
    }

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    start_time = time.time()
    for i in range(10):
        actions_list = policy.infer(example_observation)['actions']
        for actions in actions_list:
            print(f"{actions}")
    end_time = time.time()

    print(f"Time taken for 10 inferences: {end_time - start_time:.2f} seconds")
    print(f"Average time per inference: {(end_time - start_time) / 10:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test WebSocket client policy performance.")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host where the WebSocket server is running (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port where the WebSocket server is running (default: 8000)",
    )
    args = parser.parse_args()
    main(args)