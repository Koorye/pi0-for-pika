from dataclasses import dataclass

from src.deploy.pika import Pika, MultiPika
from src.deploy.piper import Piper, MultiArmPiper

@dataclass
class DeployConfig:
    host = '127.0.0.1'
    port = 8000
    frequency = 10

    # robot config
    robot_cls = Piper
    robot_cfg = dict(
        can = 'can_left',
        control_mode = 'eef_absolute',
        use_standardization = False,
    )

    # camera config
    camera_cls = Pika
    camera_cfg = dict(
        usb = '/dev/ttyUSB0',
        fisheye_camera_index = 0,
        realsense_serial_number = None,
        width = 640,
        height = 480,
        fps = 30,
        use_fisheye = True,
        use_realsense = False,
        use_realsense_depth = False,
    )
