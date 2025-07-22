import itertools
import os
import numpy as np
import shutil
from collections import defaultdict
from tqdm import tqdm

try:
    # v2.1
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.1'
except:
    # v2.0
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.0'

from ..utils.data_tools import eef_states_to_pos_rot_grip, pos_rot_grip_to_eef_states
from ..utils.image_tools import load_image
from ..utils.transforms import compute_relative_pose, robust_compute_relative_pose


def get_lerobot_default_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


class DummyDataProcessor(object):
    def __init__(self, config):
        self.config = config

        if self.config.overwrite:
            if self.config.data_root is not None:
                data_root = self.config.data_root
                if os.path.exists(data_root):
                    print(f'Overwriting data root: {data_root}? (y/n)', end=' ')
                    if input().strip().lower() != 'y':
                        print('Exiting without overwriting.')
                        return
                    shutil.rmtree(data_root, ignore_errors=True)
            else:
                data_root = get_lerobot_default_root()
                data_root = os.path.join(data_root, self.config.repo_id)
                if os.path.exists(data_root):
                    print(f'Overwriting data root: {data_root}? (y/n)', end=' ')
                    if input().strip().lower() != 'y':
                        print('Exiting without overwriting.')
                        return
                    shutil.rmtree(data_root, ignore_errors=True)
        
        self.create_dataset()

    def create_dataset(self):
        if self.config.check_only:
            print('Check only mode, skipping dataset creation.')
            return
        
        rgb_config = {
            'dtype': 'video',
            'shape': (self.config.image_height, self.config.image_width, 3),
            'names': ['height', 'width', 'channels'],
        }
        features = {rgb_name: rgb_config for rgb_name in self.config.rgb_names}
        action_len = sum(len(keys) for keys in self.config.action_keys_list)
        action_keys_flatten = list(itertools.chain.from_iterable(self.config.action_keys_list))
        features[self.config.action_name] = {
            'dtype': 'float64',
            'shape': (action_len,),
            'names': action_keys_flatten,
        }

        if self.config.use_state:
            features[self.config.state_name] = {
                'dtype': 'float64',
                'shape': (action_len,),
                'names': action_keys_flatten,
            }

        if self.config.use_depth:
            depth_config = {
                'dtype': 'uint16',
                'shape': (self.config.image_height, self.config.image_width),
                'name': ['height', 'width'],
            }
            for depth_name in self.config.depth_names:
                features[depth_name] = depth_config
        
        if self.config.data_root is not None:
            self.config.data_root = os.path.join(self.config.data_root, self.config.repo_id)
        
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            root=self.config.data_root,
            fps=self.config.fps,
            video_backend=self.config.video_backend,
            features=features,
        )
    
    def process_data(self):
        num_episodes = 3
        for episode_idx in range(num_episodes):
            print(f'Processing episode {episode_idx + 1}/{num_episodes}')
            self._add_episode('dummy')
    
    def _add_episode(self, episode_path):
        raw_outputs = self._load_episode(episode_path)
        
        if self.config.check_only:
            print(f'Check only mode, skipping adding episode {episode_path}')
            return

        raw_images = raw_outputs['raw_images']
        raw_actions = raw_outputs['raw_actions']
        instruction = raw_outputs['instruction']
        if self.config.use_depth:
            raw_depths = raw_outputs['raw_depths']
        
        indexs = list(range(len(raw_images[self.config.rgb_names[0]])))
        
        for i in tqdm(indexs[:-1], desc=f'Adding episode {episode_path}'):
            states = np.concatenate([raw_actions[action_dir][i] for action_dir in self.config.action_dirs])
            actions = np.concatenate([raw_actions[action_dir][i + 1] for action_dir in self.config.action_dirs])
            if not self._check_nonoop_actions(states, actions):
                print(f'Skipping frame {i} due to non-noop actions.')
                continue

            if self.config.use_delta:
                pos_rot_grips = eef_states_to_pos_rot_grip(states)
                next_pos_rot_grips = eef_states_to_pos_rot_grip(actions)
                
                pos_rot_grip_deltas = []
                for pos_rot_grip, next_pos_rot_grip in zip(pos_rot_grips, next_pos_rot_grips):
                    pos_delta, rot_delta = compute_relative_pose(pos_rot_grip[0], pos_rot_grip[1], 
                                                                 next_pos_rot_grip[0], next_pos_rot_grip[1])
                    pos_rot_grip_deltas.append((pos_delta, rot_delta, next_pos_rot_grip[2]))
                actions = pos_rot_grip_to_eef_states(pos_rot_grip_deltas)

            frame = {rgb_name: load_image(raw_images[rgb_name][i]) for rgb_name in self.config.rgb_names}
            frame[self.config.action_name] = actions

            if self.config.use_state:
                frame[self.config.state_name] = states

            if self.config.use_depth:
                frame.update({depth_name: load_image(raw_depths[depth_name][i]) 
                              for depth_name in self.config.depth_names})

            if _LEROBOT_VERSION == '2.0':
                self.dataset.add_frame(frame)
            elif _LEROBOT_VERSION == '2.1':
                self.dataset.add_frame(frame, task=instruction)
            else:
                raise ValueError(f'Unsupported LeRobot version: {_LEROBOT_VERSION}')
            
        if _LEROBOT_VERSION == '2.0':
            self.dataset.save_episode(task=instruction)
        elif _LEROBOT_VERSION == '2.1':
            self.dataset.save_episode()
        else:
            raise ValueError(f'Unsupported LeRobot version: {_LEROBOT_VERSION}')
        
    def _load_episode(self, episode_path):
        num_frames_per_episode = 100

        raw_images = defaultdict(list)
        raw_actions = defaultdict(list)
        instruction = 'do something'
        
        for frame_idx in range(num_frames_per_episode):
            for rgb_name in self.config.rgb_names:
                image = np.random.randint(0, 255, (self.config.image_height, self.config.image_width, 3), dtype=np.uint8)
                raw_images[rgb_name].append(image)
            
            for action_dir, action_keys in zip(self.config.action_dirs, self.config.action_keys_list):
                action_data = np.random.rand(len(action_keys))
                raw_actions[action_dir].append(action_data)
        
        outputs = {
            'raw_images': raw_images,
            'raw_actions': raw_actions,
            'instruction': instruction
        }

        if self.config.use_depth:
            raw_depths = defaultdict(list)
            for frame_idx in range(num_frames_per_episode):
                for depth_name in self.config.depth_names:
                    depth_image = np.random.randint(0, 65535, (self.config.image_height, self.config.image_width), dtype=np.uint16)
                    raw_depths[depth_name].append(depth_image)
            outputs['raw_depths'] = raw_depths
        
        return outputs
    
    def _check_nonoop_actions(self, states, actions):
        states = eef_states_to_pos_rot_grip(states)
        actions = eef_states_to_pos_rot_grip(actions)
        
        for state, action in zip(states, actions):
            position_diff = np.linalg.norm(np.array(state[0]) - np.array(action[0]))
            rotation_diff = np.linalg.norm(np.array(state[1]) - np.array(action[1]))
            if (
                position_diff > self.config.position_nonoop_threshold or
                rotation_diff > self.config.rotation_nonoop_threshold
            ):
                return True
        
        return False
