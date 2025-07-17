from dataclasses import dataclass

from src.deploy.dummy_camera import DummyCamera, MultiDummyCamera
from src.deploy.dummy_robot import DummyRobot, MultiArmDummyRobot
from src.deploy.pika import Pika, MultiPika
from src.deploy.piper import Piper, MultiArmPiper

@dataclass
class DeployConfig:
    host = '127.0.0.1'
    port = 8000
    frequency = 10

    # robot config
    robot_cls = Piper
    # robot_cls = DummyRobot
    robot_cfg = dict(
        can = 'can_left',
        control_mode = 'eef_delta_gripper',
        # control_mode = 'eef_delta_root',
        use_standardization = True,
    )

    # camera config
    # camera_cls = Pika
    camera_cls = DummyCamera
    camera_cfg = dict(
        usb = '/dev/ttyUSB1',
        fisheye_camera_index = 2,
        realsense_serial_number = None,
        width = 640,
        height = 480,
        fps = 30,
        use_fisheye = True,
        use_realsense = False,
        use_realsense_depth = False,
    )
