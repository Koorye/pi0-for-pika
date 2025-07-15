from .dummy_client import DummyClient


class Client(DummyClient):
    def __init__(
        self,
        camera, 
        robot, 
        prompt,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.camera = camera
        self.robot = robot
        self.robot.reset()
    
    def do_action(self):
        obs = self.camera.get_observation()
        obs.update(self.robot.get_observation())
        obs['prompt'] = self.prompt
        # rgb = obs['left_wrist_fisheye_rgb']
        # import imageio
        # imageio.imwrite('test.jpg', rgb)
        actions = self.policy.infer(obs)['actions'][:4]
        if len(actions.shape) == 1:
            actions = [actions]
        for action in actions:
            self.robot.do_action(action)
