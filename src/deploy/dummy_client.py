import numpy as np
import time
from openpi_client import websocket_client_policy


class DummyClient(object):
    def __init__(
        self, 
        host='127.0.0.1', 
        port=8000, 
        frequency=10,
    ):
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
        )
        self.frequency = frequency
        
        self.example_observation = {
            'rgb': np.zeros((480, 640, 3), dtype=np.uint8),
            'state': np.zeros(7, dtype=np.float32),
        }
    
    def run(self):
        while True:
            self.do_action()
            time.sleep(1 / self.frequency)
    
    def do_action(self):
        action = self.policy.infer(self.example_observation)['action']
        print(action)
