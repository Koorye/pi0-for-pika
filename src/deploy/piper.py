import numpy as np
import time
from piper_sdk import C_PiperInterface_V2

from .standardlization import get_standardization
from .utils import delta_to_absolute_root_translation, delta_to_absolute_gripper_translation

from .dummy_robot import DummyRobot, MultiArmDummyRobot


_INIT_STATES = [100000, 0, 300000, 180000, 90000, 180000, 60000]
_INIT_JOINT_STATES = [0, 22255, -44503, 0, 27283, 0]


class Piper(DummyRobot):
    def __init__(self, 
                 can, 
                 control_mode='eef_absolute',
                 use_standardization=True):
        super().__init__(can, control_mode, use_standardization)
        
        self.piper = C_PiperInterface_V2(can)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
    
    def get_eef_states(self):
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        x, y, z, rx, ry, rz = end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, \
                              end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis
        grip = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        return np.array([x, y, z, rx, ry, rz, grip])
    
    def set_eef_states(self, states):
        self.piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
        x, y, z, rx, ry, rz, grip = states[:7]
        x, y, z, rx, ry, rz, grip = int(x), int(y), int(z), int(rx), int(ry), int(rz), int(grip)
        self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)
        self.piper.GripperCtrl(grip, 1000, 0x01, 0)
        self.states_list.append(self.get_eef_states())
    
    def set_joint_states(self, states):
        self.piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        j1, j2, j3, j4, j5, j6 = states[:6]
        j1, j2, j3, j4, j5, j6 = int(j1), int(j2), int(j3), int(j4), int(j5), int(j6)
        self.piper.JointCtrl(j1, j2, j3, j4, j5, j6)

    def reset(self):
        self.set_eef_states(_INIT_STATES)
        for _ in range(100):
            time.sleep(0.01)
        # self.set_joint_states(_INIT_JOINT_STATES)
        # for _ in range(100):
        #     time.sleep(0.01)
    
    def stop(self):
        while self.piper.DisablePiper():
            time.sleep(0.01)


class MultiArmPiper(MultiArmDummyRobot):
    def __init__(self, 
                 can_left, 
                 can_right, 
                 control_mode='eef_absolute',
                 use_standardization=True):
        super().__init__(can_left, can_right, control_mode, use_standardization)
        self.left_piper = Piper(can_left, control_mode, use_standardization)
        self.right_piper = Piper(can_right, control_mode, use_standardization)
    
    def get_observation(self):
        left_obs = self.left_piper.get_observation()
        right_obs = self.right_piper.get_observation()
        states = np.concatenate((left_obs['states'], right_obs['states']))
        return {
            'states': states,
        }
    
    def do_action(self, action):
        left_action = action[:7]
        right_action = action[7:]
        self.left_piper.do_action(left_action)
        self.right_piper.do_action(right_action)
    
    def get_eef_state(self):
        left_state = self.left_piper.get_eef_state()
        right_state = self.right_piper.get_eef_state()
        return np.concatenate((left_state, right_state))
    
    def set_eef_state(self, action):
        left_action = action[:7]
        right_action = action[7:]
        self.left_piper.set_eef_state(left_action)
        self.right_piper.set_eef_state(right_action)
    
    def reset(self):
        self.left_piper.reset()
        self.right_piper.reset()

    def stop(self):
        self.left_piper.stop()
        self.right_piper.stop()

    def visualize(self):
        left_positions = [state[0:3] for state in self.left_piper.states_list]
        right_positions = [state[0:3] for state in self.right_piper.states_list]
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