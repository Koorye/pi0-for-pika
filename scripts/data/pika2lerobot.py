import sys
sys.path.append('.')

import argparse
import importlib

from data.pika_data_processor import DataProcessor


def main(args):
    config = importlib.import_module('scripts.data.' + args.config).DataConfig()
    processor = DataProcessor(config)
    processor.process_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Pika2LeRobot data.")
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Path to the configuration module.",
    )
    args = parser.parse_args()
    main(args)
