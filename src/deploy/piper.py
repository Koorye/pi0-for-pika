import numpy as np
import time
from piper_sdk import C_PiperInterface_V2


class Camera(object):
    def __init__(self, can):
        pass
    
    def get_observation(self):
        pass


class Piper(object):
    def __init__(self, can):
        self.piper = C_PiperInterface_V2(can)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        print('Piper init finished')
    
    def get_observation(self):
        # TODO: Implement camera observation
        state = self.get_eef_state()
        return {
            'state': state,
        }
    
    def get_eef_state(self):
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        x, y, z, rx, ry, rz = end_pose.X, end_pose.Y, end_pose.Z, end_pose.RX, end_pose.RY, end_pose.RZ
        grip = self.piper.GetArmGripperMsgs()
        return np.array([x, y, z, rx, ry, rz, grip])
    
    def set_eef_action(self, action):
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        x, y, z, rx, ry, rz, grip = action
        self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)
        self.piper.GripperCtrl(grip, 1000, 0x01, 0)


class MultiArmPiper(object):
    def __init__(self, left_can, right_can):
        self.left_piper = Piper(left_can)
        self.right_piper = Piper(right_can)
    
    def get_observation(self):
        left_obs = self.left_piper.get_observation()
        right_obs = self.right_piper.get_observation()
        state = np.concatenate((left_obs['state'], right_obs['state']))
        return {
            'state': state,
        }
    
    def get_eef_state(self):
        left_state = self.left_piper.get_eef_state()
        right_state = self.right_piper.get_eef_state()
        return np.concatenate((left_state, right_state))
    
    def set_eef_action(self, action):
        left_action = action[:7]
        right_action = action[7:]
        self.left_piper.set_eef_action(left_action)
        self.right_piper.set_eef_action(right_action)
