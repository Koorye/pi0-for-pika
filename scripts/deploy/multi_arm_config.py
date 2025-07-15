from dataclasses import dataclass

from src.deploy.pika import Pika, MultiPika
from src.deploy.piper import Piper, MultiArmPiper

@dataclass
class DeployConfig:
    host = '127.0.0.1'
    port = 8000
    frequency = 10
    prompt = 'pick the banana in the basket'

    # robot config
    robot_cls = MultiArmPiper
    robot_cfg = dict(
        can_left = 'can_left',
        can_right = 'can_right',
        control_mode = 'eef_delta_gripper',
        use_standardization = True,
    )

    # camera config
    camera_cls = MultiPika
    camera_cfg = dict(
        usb_left = '/dev/ttyUSB81',
        usb_right = '/dev/ttyUSB82',
        fisheye_camera_index_left = 81,
        fisheye_camera_index_right = 82,
        realsense_serial_number_left = None,
        realsense_serial_number_right = None,
        width = 640,
        height = 480,
        fps = 30,
        use_fisheye = True,
        use_realsense = False,
        use_realsense_depth = False,
    )
