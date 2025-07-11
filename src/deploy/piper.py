import numpy as np
import time
from piper_sdk import C_PiperInterface_V2

from .utils import delta_to_absolute_root_translation, delta_to_absolute_gripper_translation


_INIT_STATE = [57000, 0, 300000, 0, 90000, 0, 0, 60000]


class Camera(object):
    def __init__(self, can):
        pass
    
    def get_observation(self):
        pass


class Piper(object):
    def __init__(self, can, control_mode='eef_absolute'):
        self.can = can
        self.control_mode = control_mode
        
        self.piper = C_PiperInterface_V2(can)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)

        self.reset()
        for _ in range(100):
            time.sleep(0.01)
        
        print('Piper init finished')
    
    def get_observation(self):
        # TODO: Implement camera observation
        state = self.get_eef_state()
        return {
            'state': state,
        }
    
    def do_action(self, action):
        if self.control_mode == 'eef_absolute':
            self.set_eef_state(action)
        elif self.control_mode == 'eef_delta_root':
            state = self.get_eef_state()
            self.set_eef_state(delta_to_absolute_root_translation(action, state))
        elif self.control_mode == 'eef_delta_gripper':
            state = self.get_eef_state()
            self.set_eef_state(delta_to_absolute_gripper_translation(action, state))
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
    
    def get_eef_state(self):
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        x, y, z, rx, ry, rz = end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, \
                              end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis
        grip = self.piper.GetArmGripperMsgs()
        return np.array([x, y, z, rx, ry, rz, grip])
    
    def set_eef_state(self, state):
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        x, y, z, rx, ry, rz, grip = state
        self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)
        self.piper.GripperCtrl(grip, 1000, 0x01, 0)
    
    def reset(self):
        self.set_eef_state(_INIT_STATE)


class MultiArmPiper(object):
    def __init__(self, 
                 left_can, 
                 right_can, 
                 control_mode='eef_absolute'):
        self.left_piper = Piper(left_can, control_mode)
        self.right_piper = Piper(right_can, control_mode)
    
    def get_observation(self):
        left_obs = self.left_piper.get_observation()
        right_obs = self.right_piper.get_observation()
        state = np.concatenate((left_obs['state'], right_obs['state']))
        return {
            'state': state,
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
