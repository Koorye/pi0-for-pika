import math
from dataclasses import dataclass


@dataclass
class DataConfig:
    overwrite = True
    check_only = False

    source_data_roots = [
        # 'examples/pika_example_data',
        '/home/shihanwu/Datasets/pika-demo',
    ]

    image_height = 480
    image_width = 640
    rgb_dirs = [
        'camera/color/camera_realsense_c',
        'camera/color/pikaDepthCamera_l',
        'camera/color/pikaFisheyeCamera_l',
        'camera/color/pikaDepthCamera_r',
        'camera/color/pikaFisheyeCamera_r',
    ]
    rgb_names = [
        'observation.images.cam_third',
        'observation.images.cam_left_wrist',
        'observation.images.cam_left_wrist_fisheye',
        'observation.images.cam_right_wrist',
        'observation.images.cam_right_wrist_fisheye',
    ]

    use_depth = True
    depth_dirs = [
        'camera/depth/pikaDepthCamera_c',
        'camera/depth/pikaDepthCamera_l',
        'camera/depth/pikaDepthCamera_r',
    ]
    depth_names = [
        'observation.depths.cam_third',
        'observation.depths.cam_left_wrist',
        'observation.depths.cam_right_wrist',
    ]

    action_name = 'action'
    action_dirs = [
        'localization/pose/pika_l',
        'gripper/encoder/pika_l',
        'localization/pose/pika_r',
        'gripper/encoder/pika_r',
    ]
    action_keys_list = [
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
    ]
    position_nonoop_threshold = 1e-2 # 1cm
    rotation_nonoop_threshold = math.pi / 180 # 1 degree
    use_delta = True

    use_state = True
    state_name = 'observation.state'

    instruction_path = 'instructions.json'
    default_instruction = 'do something'

    repo_id = 'Koorye/pika-demo'
    data_root = None
    fps = 10
    video_backend = 'pyav'
