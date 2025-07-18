import numpy as np


class DummyCamera(object):
    def __init__(self, 
                 width=640,
                 height=480,
                 fps=30,
                 use_fisheye=True,
                 use_realsense=True,
                 use_realsense_depth=False,
                 prefix='left_wrist'):
        self.width = width
        self.height = height
        self.fps = fps
        self.use_fisheye = use_fisheye
        self.use_realsense = use_realsense
        self.use_realsense_depth = use_realsense_depth
        self.prefix = prefix

    def get_observation(self):
        outputs = {}
        if self.use_fisheye:
            outputs[f'{self.prefix}_fisheye_rgb'] = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        if self.use_realsense:
            outputs[f'{self.prefix}_base_rgb'] = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        if self.use_realsense_depth:
            outputs[f'{self.prefix}_base_depth'] = np.random.randint(0, 2 ** 16, (self.height, self.width), dtype=np.uint16)
        return outputs
