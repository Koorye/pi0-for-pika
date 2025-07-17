import numpy as np


class DummyCamera(object):
    def __init__(self, 
                 usb,
                 fisheye_camera_index=None,
                 realsense_serial_number=None,
                 width=640,
                 height=480,
                 fps=30,
                 use_fisheye=True,
                 use_realsense=True,
                 use_realsense_depth=False,
                 prefix='left_wrist'):
        self.width = width
        self.height = height
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


class MultiDummyCamera(object):
    def __init__(self, 
                 usb_left, 
                 usb_right, 
                 fisheye_camera_index_left=None, 
                 fisheye_camera_index_right=None,
                 realsense_serial_number_left=None,
                 realsense_serial_number_right=None,
                 width=640,
                 height=480,
                 fps=30,
                 use_fisheye=True,
                 use_realsense=True,
                 use_realsense_depth=False):
        self.left_dummy_camera = DummyCamera(
            usb_left, 
            fisheye_camera_index=fisheye_camera_index_left, 
            realsense_serial_number=realsense_serial_number_left,
            width=width, 
            height=height, 
            fps=fps,
            use_fisheye=use_fisheye,
            use_realsense=use_realsense,
            use_realsense_depth=use_realsense_depth,
            prefix='left_wrist'
        )
        self.right_dummy_camera = DummyCamera(
            usb_right, 
            fisheye_camera_index=fisheye_camera_index_right, 
            realsense_serial_number=realsense_serial_number_right,
            width=width, 
            height=height, 
            fps=fps,
            use_fisheye=use_fisheye,
            use_realsense=use_realsense,
            use_realsense_depth=use_realsense_depth,
            prefix='right_wrist'
        )
    
    def get_observation(self):
        left_obs = self.left_dummy_camera.get_observation()
        right_obs = self.right_dummy_camera.get_observation()
        return {**left_obs, **right_obs}