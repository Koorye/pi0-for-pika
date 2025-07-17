import sys
sys.path.append('.')

import argparse
import importlib
import time

from src.deploy.client import Client


def main(args):
    config = importlib.import_module('scripts.deploy.' + args.config).DeployConfig()
    client = Client(
        camera=config.camera_cls(**config.camera_cfg),
        robot=config.robot_cls(**config.robot_cfg),
        host=config.host,
        port=config.port,
        frequency=config.frequency,
        prompt=config.robot_cfg.get('prompt', 'do something')  # Default prompt
    )

    time.sleep(3)
    input('Press Enter to start the client...')
    client.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy the OpenPI client.')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config',
        help='Path to the configuration file.'
    )
    args = parser.parse_args()
    main(args)
    