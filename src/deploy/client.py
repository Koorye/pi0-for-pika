import time

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

    def run(self):
        while True:
            self.do_action()
            self.count += 1
            if self.count % 50 == 0:
                self.robot.visualize()
            time.sleep(1 / self.frequency)
    
    def do_action(self):
        obs = self.camera.get_observation()
        obs.update(self.robot.get_observation())
        obs['prompt'] = self.prompt
        
        # left_rgb = obs['left_wrist_fisheye_rgb']
        # right_rgb = obs['right_wrist_fisheye_rgb']
        # rgb = np.concatenate([left_rgb, right_rgb], axis=1)
        # imageio.imwrite('test.jpg', rgb)
        actions_list = self.policy.infer(obs)['actions'][:4]

        for actions in actions_list:
            print(actions[:6])
            self.robot.do_action(actions)
