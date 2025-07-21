from pika import sense

from .dummy_camera import DummyCamera


class PikaCamera(DummyCamera):
    def __init__(
            self, 
            usb,
            fisheye_camera_index=None,
            realsense_serial_number=None,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        _sense = sense(usb)
    
        if not _sense.connect():
            print("Pika Sense connect failed")
            return
    
        _sense.set_camera_param(self.width, self.height, self.fps)

        if self.use_fisheye:
            if fisheye_camera_index is None:
                raise ValueError("Fisheye camera index must be provided")
            _sense.set_fisheye_camera_index(fisheye_camera_index)
            self.fisheye_camera = _sense.get_fisheye_camera()
        
        if self.use_realsense:
            _sense.set_realsense_serial_number(realsense_serial_number)
            if realsense_serial_number is None:
                raise ValueError("Realsense serial number must be provided")
            self.realsense_camera = _sense.get_realsense_camera()
    
    def get_observation(self):
        outputs = {}

        if self.use_fisheye:
            success, frame = self.fisheye_camera.get_frame()
            frame = frame[:, :, ::-1]  # Convert BGR to RGB
            if not success:
                print("Failed to get fisheye frame")
                return None
            outputs[f'{self.prefix}_fisheye_rgb'] = frame
        
        if self.use_realsense:
            success, color_frame = self.realsense_camera.get_color_frame()
            color_frame = color_frame[:, :, ::-1]  # Convert BGR to RGB
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
