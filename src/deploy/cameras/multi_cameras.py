class MultiCameras(object):
    def __init__(
            self, 
            left_camera,
            right_camera,
        ):
        self.left_camera = left_camera
        self.right_camera = right_camera
    
    def get_observation(self):
        left_obs = self.left_camera.get_observation()
        right_obs = self.right_camera.get_observation()
        return {**left_obs, **right_obs}
