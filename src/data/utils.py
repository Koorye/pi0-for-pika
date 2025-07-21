import imageio.v3 as imageio
import numpy as np
import os


def get_lerobot_default_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


def load_image(image_path):
    if isinstance(image_path, np.ndarray):
        return image_path
    return imageio.imread(image_path)
