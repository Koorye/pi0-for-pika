from .dummy_client import DummyClient


class Client(DummyClient):
    def __init__(
        self, 
        piper, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.piper = piper
    
    def do_action(self):
        obs = self.piper.get_observation()
        action = self.policy.infer(obs)['action']
        self.piper.do_action(action)
