import numpy as np

from .standardlization import get_standardization
from .utils import delta_to_absolute_root_translation, delta_to_absolute_gripper_translation


_INIT_STATES = [100000, 0, 300000, 0, 0, 0, 60000]
_INIT_JOINT_STATES = [0, 22255, -44503, 0, 27283, 0]


class DummyRobot(object):
    def __init__(self, 
                 can, 
                 control_mode='eef_absolute',
                 use_standardization=True,
                 init_states=_INIT_STATES):
        self.can = can
        self.control_mode = control_mode
        self.use_standardization = use_standardization
        self.init_states = init_states

        transforms = get_standardization('piper')
        self.input_transform = transforms['input']
        self.output_transform = transforms['output']

        self.states_list = []
    
    def get_observation(self):
        states = self.get_eef_states()
        if self.use_standardization:
            states = self.input_transform(states)
        return {
            'states': states,
        }
    
    def do_action(self, actions):
        if self.use_standardization:
            actions = self.output_transform(actions)
        
        if self.control_mode == 'eef_absolute':
            self.set_eef_states(actions)
        elif self.control_mode == 'eef_delta_root':
            states = self.get_eef_states()
            self.set_eef_states(delta_to_absolute_root_translation(actions, states))
        elif self.control_mode == 'eef_delta_gripper':
            states = self.get_eef_states()
            self.set_eef_states(delta_to_absolute_gripper_translation(actions, states))
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
    
    def get_eef_states(self):
        return self.states
    
    def set_eef_states(self, states):
        self.states = states
        self.states_list.append(states)

    def reset(self):
        self.states = self.init_states
    
    def stop(self):
        pass

    def visualize(self):
        left_positions = [state[0:3] for state in self.states_list]
        # scatter 3d
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        left_positions = np.array(left_positions)
        ax.scatter(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2], c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.axis('equal')
        plt.title(self.can)
        plt.show()


class MultiArmDummyRobot(object):
    def __init__(self, 
                 can_left, 
                 can_right, 
                 control_mode='eef_absolute',
                 use_standardization=True):
        self.left_dummy_robot = DummyRobot(can_left, control_mode, use_standardization, _INIT_STATES)
        right_init_states = _INIT_STATES.copy()
        right_init_states[1] -= 500000  # Offset for right arm
        self.right_dummy_robot = DummyRobot(can_right, control_mode, use_standardization, right_init_states)
    
    def get_observation(self):
        left_obs = self.left_dummy_robot.get_observation()
        right_obs = self.right_dummy_robot.get_observation()
        states = np.concatenate((left_obs['states'], right_obs['states']))
        return {
            'states': states,
        }
    
    def do_action(self, action):
        # left_action = action[:7]
        # right_action = action[7:]
        left_action = action[7:]
        right_action = action[:7]
        self.left_dummy_robot.do_action(left_action)
        self.right_dummy_robot.do_action(right_action)
    
    def get_eef_state(self):
        left_state = self.left_dummy_robot.get_eef_state()
        right_state = self.right_dummy_robot.get_eef_state()
        return np.concatenate((left_state, right_state))
    
    def set_eef_state(self, action):
        left_action = action[:7]
        right_action = action[7:]
        self.left_dummy_robot.set_eef_state(left_action)
        self.right_dummy_robot.set_eef_state(right_action)
    
    def reset(self):
        self.left_dummy_robot.reset()
        self.right_dummy_robot.reset()

    def stop(self):
        self.left_dummy_robot.stop()
        self.right_dummy_robot.stop()

    def visualize(self):
        left_positions = [state[0:3] for state in self.left_dummy_robot.states_list]
        right_positions = [state[0:3] for state in self.right_dummy_robot.states_list]
        # scatter 3d
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        left_positions = np.array(left_positions)
        ax.scatter(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2], c='r', marker='o')
        right_positions = np.array(right_positions)
        ax.scatter(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2], c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.axis('equal')
        plt.show()