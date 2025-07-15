from pika import sense


class Pika(object):
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
        self.use_fisheye = use_fisheye
        self.use_realsense = use_realsense
        self.use_realsense_depth = use_realsense_depth
        self.prefix = prefix

        _sense = sense(usb)
    
        if not _sense.connect():
            print("Pika Sense connect failed")
            return
    
        _sense.set_camera_param(width, height, fps)

        if use_fisheye:
            if fisheye_camera_index is None:
                raise ValueError("Fisheye camera index must be provided")
            _sense.set_fisheye_camera_index(fisheye_camera_index)
            self.fisheye_camera = _sense.get_fisheye_camera()
        
        if use_realsense:
            _sense.set_realsense_serial_number(realsense_serial_number)
            if realsense_serial_number is None:
                raise ValueError("Realsense serial number must be provided")
            self.realsense_camera = _sense.get_realsense_camera()
    
    def get_observation(self):
        outputs = {}

        if self.use_fisheye:
            success, frame = self.fisheye_camera.get_frame()
            if not success:
                print("Failed to get fisheye frame")
                return None
            outputs[f'{self.prefix}_fisheye_rgb'] = frame
        
        if self.use_realsense:
            success, color_frame = self.realsense_camera.get_color_frame()
            if not success:
                print("Failed to get realsense frame")
                return None
            outputs[f'{self.prefix}_base_rgb'] = color_frame
        
        if self.use_realsense_depth:
            success, depth_frame = self.realsense_camera.get_depth_frame()
            if not success:
                print("Failed to get realsense depth frame")
                return None
            outputs[f'{self.prefix}_base_depth'] = depth_frame
        
        return outputs


class MultiPika(object):
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
        self.left_pika = Pika(
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
        self.right_pika = Pika(
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
        left_obs = self.left_pika.get_observation()
        right_obs = self.right_pika.get_observation()
        return {**left_obs, **right_obs}