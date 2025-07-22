import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.1'
except:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.0'


class PikaTrajectoryLoader:
    def __init__(
            self, 
            init_position, 
            init_euler, 
            init_grip,
        ):
        self.init_position = init_position
        self.init_euler = init_euler
        self.init_grip = init_grip

        self.T_world_current = np.eye(4)
        self.T_world_current[:3, 3] = init_position
        rot = Rotation.from_euler('xyz', init_euler)
        self.T_world_current[:3, :3] = rot.as_matrix()

        self.trajectory = {
            'timesteps': [],
            'positions': [],
            'euler_angles': [],
            'grips': [],
        }
        self.timestep = 0

    def add_frame(self, position, euler, grip):
        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append(position)
        self.trajectory['euler_angles'].append(euler)
        self.trajectory['grips'].append(grip)
        self.timestep += 1
    
    def add_delta_frame(self, position, euler, grip):
        dx, dy, dz = position
        drx, dry, drz = euler
        
        T_rel = np.eye(4)
        T_rel[:3, 3] = [dx, dy, dz]
        
        rel_rotation = Rotation.from_euler('xyz', [drx, dry, drz])
        rel_rot_matrix = rel_rotation.as_matrix()
        T_rel[:3, :3] = rel_rot_matrix @ T_rel[:3, :3]
        
        self.T_world_current = self.T_world_current @ T_rel
        
        global_position = self.T_world_current[:3, 3].copy()
        global_rot_matrix = self.T_world_current[:3, :3].copy()
        
        global_euler = Rotation.from_matrix(global_rot_matrix).as_euler('xyz')
        
        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append(global_position.tolist())
        self.trajectory['euler_angles'].append(global_euler.tolist())
        self.trajectory['grips'].append(grip)
        self.timestep += 1

    def get_trajectory(self):
        return self.trajectory
    
    def reset(self, init_position, init_euler, init_grip):
        self.__init__(
            init_position=init_position, 
            init_euler=init_euler, 
            init_grip=init_grip
        )
    
    def plot_trajectory_3d(self, figsize=(10, 8)):
        positions = np.array(self.trajectory['positions'])
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2)
        
        ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 
                'go', markersize=8, label='start')
        ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                'ro', markersize=8, label='end')
        
        n_points = len(positions)
        step = max(1, n_points // 20) 
        
        euler_angles = np.array(self.trajectory['euler_angles'])
        
        for i in range(0, n_points, step):
            if i < n_points:
                pos = positions[i]
                direction = Rotation.from_euler('xyz', euler_angles[i]).apply([1, 0, 0])
                ax.quiver(pos[0], pos[1], pos[2], 
                         direction[0], direction[1], direction[2],
                         length=0.1, color='r', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.tight_layout()
        plt.axis('equal')
        plt.show()


class MultiPikaTrajectoryLoader:
    def __init__(
            self, 
            init_positions, 
            init_eulers, 
            init_grips, 
            names,
        ):
        self.names = names
        self.converters = []
        for name, pos, euler, grip in zip(names, init_positions, init_eulers, init_grips):
            converter = PikaTrajectoryLoader(pos, euler, grip)
            self.converters.append(converter)

    def add_frame(self, positions, eulers, grips):
        for converter, pos, euler, grip in zip(self.converters, positions, eulers, grips):
            converter.add_frame(pos, euler, grip)

    def add_delta_frame(self, positions, eulers, grips):
        for converter, pos, euler, grip in zip(self.converters, positions, eulers, grips):
            converter.add_delta_frame(pos, euler, grip)
    
    def get_trajectories(self):
        trajectories = {}
        for name, converter in zip(self.names, self.converters):
            trajectories[name] = converter.get_trajectory()
        return trajectories
    
    def reset(self, init_positions, init_eulers, init_grips):
        for converter, pos, euler, grip in zip(self.converters, init_positions, init_eulers, init_grips):
            converter.reset(pos, euler, grip)

    def plot_trajectories_3d(self, figsize=(10, 8), names=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        for name, converter in zip(self.names, self.converters):
            if names is not None:
                if name not in names:
                    continue
            
            positions = np.array(converter.trajectory['positions'])
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    label=name, linewidth=2)
            
            ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 
                    'go', markersize=8)
            ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                    'ro', markersize=8)
            
            n_points = len(positions)
            step = max(1, n_points // 20) 
            euler_angles = np.array(converter.trajectory['euler_angles'])

            for i in range(0, n_points, step):
                if i < n_points:
                    pos = positions[i]
                    direction = Rotation.from_euler('xyz', euler_angles[i]).apply([1, 0, 0])
                    ax.quiver(pos[0], pos[1], pos[2], 
                             direction[0], direction[1], direction[2],
                             length=0.1, color='r', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.tight_layout()
        plt.axis('equal')
        plt.show()

def compute_relative_pose(prev_pos, prev_euler, curr_pos, curr_euler, rotation_order='xyz'):
    """
    计算当前帧相对于上一帧的相对位置和旋转变化
    
    参数:
        prev_state: 上一帧状态 [x, y, z, rx, ry, rz]
        current_state: 当前帧状态 [x, y, z, rx, ry, rz]
        rotation_order: 欧拉角的旋转顺序 ('xyz', 'zyx'等)
    
    返回:
        relative_pose: 相对位姿变化 [dx, dy, dz, drx, dry, drz]
    """
    # 1. 构建上一帧的齐次变换矩阵
    T_prev = np.eye(4)
    T_prev[:3, 3] = prev_pos
    T_prev[:3, :3] = Rotation.from_euler(rotation_order, prev_euler).as_matrix()
    
    # 2. 构建当前帧的齐次变换矩阵
    T_curr = np.eye(4)
    T_curr[:3, 3] = curr_pos
    T_curr[:3, :3] = Rotation.from_euler(rotation_order, curr_euler).as_matrix()
    
    # 3. 计算相对变换矩阵 (在上一帧坐标系中)
    T_relative = np.linalg.inv(T_prev) @ T_curr
    
    # 4. 提取相对位置变化
    dx, dy, dz = T_relative[:3, 3]
    
    # 5. 提取相对旋转变化
    relative_rot = Rotation.from_matrix(T_relative[:3, :3])
    drx, dry, drz = relative_rot.as_euler(rotation_order)
    
    return np.array([dx, dy, dz]), np.array([drx, dry, drz])

# 增强版：处理角度跳变问题
def robust_compute_relative_pose(prev_pos, prev_euler, curr_pos, curr_euler, rotation_order='xyz'):
    """
    鲁棒的相对位姿计算，处理角度跳变问题
    
    参数:
        prev_state: 上一帧状态 [x, y, z, rx, ry, rz]
        current_state: 当前帧状态 [x, y, z, rx, ry, rz]
        rotation_order: 欧拉角的旋转顺序 ('xyz', 'zyx'等)
    
    返回:
        relative_pose: 相对位姿变化 [dx, dy, dz, drx, dry, drz]
    """
    # 1. 构建上一帧的齐次变换矩阵
    rot_prev = Rotation.from_euler(rotation_order, prev_euler)
    T_prev = np.eye(4)
    T_prev[:3, 3] = prev_pos
    T_prev[:3, :3] = rot_prev.as_matrix()
    
    # 2. 构建当前帧的齐次变换矩阵
    rot_curr = Rotation.from_euler(rotation_order, curr_euler)
    T_curr = np.eye(4)
    T_curr[:3, 3] = curr_pos
    T_curr[:3, :3] = rot_curr.as_matrix()
    
    # 3. 计算相对变换矩阵 (在上一帧坐标系中)
    T_relative = np.linalg.inv(T_prev) @ T_curr
    
    # 4. 提取相对位置变化
    dx, dy, dz = T_relative[:3, 3]
    
    # 5. 使用四元数计算相对旋转变化（避免角度跳变）
    quat_prev = rot_prev.as_quat()
    quat_curr = rot_curr.as_quat()
    
    # 计算相对四元数: q_rel = q_prev⁻¹ ⊗ q_curr
    q_prev_inv = np.array([quat_prev[3], -quat_prev[0], -quat_prev[1], -quat_prev[2]])  # [w, -x, -y, -z]
    q_rel = np.array([
        q_prev_inv[0]*quat_curr[3] - q_prev_inv[1]*quat_curr[0] - q_prev_inv[2]*quat_curr[1] - q_prev_inv[3]*quat_curr[2],
        q_prev_inv[0]*quat_curr[0] + q_prev_inv[1]*quat_curr[3] + q_prev_inv[2]*quat_curr[2] - q_prev_inv[3]*quat_curr[1],
        q_prev_inv[0]*quat_curr[1] - q_prev_inv[1]*quat_curr[2] + q_prev_inv[2]*quat_curr[3] + q_prev_inv[3]*quat_curr[0],
        q_prev_inv[0]*quat_curr[2] + q_prev_inv[1]*quat_curr[1] - q_prev_inv[2]*quat_curr[0] + q_prev_inv[3]*quat_curr[3]
    ])
    
    # 将相对四元数转换为欧拉角
    relative_rot = Rotation.from_quat([q_rel[1], q_rel[2], q_rel[3], q_rel[0]])  # [x, y, z, w]
    drx, dry, drz = relative_rot.as_euler(rotation_order)
    
    return np.array([dx, dy, dz]), np.array([drx, dry, drz])


def main(args):
    dataset = LeRobotDataset(args.repo_id)
    current_episode = None
    converter = None
    past_positions = None
    past_eulers = None

    for sample in tqdm(dataset, desc="Processing samples"):
        episode = sample['episode_index']
        state = sample['observation.state'].numpy()
        action = sample['action'].numpy()

        if episode != current_episode:
            current_episode = episode
            left_position, left_euler, left_grip = state[:3], state[3:6], state[6]
            right_position, right_euler, right_grip = state[7:10], state[10:13], state[13]

            if converter is not None:
                converter.plot_trajectories_3d(names=['left_gripper', 'right_gripper'])
            
            converter = MultiPikaTrajectoryLoader(
                init_positions=[left_position, right_position],
                init_eulers=[left_euler, right_euler],
                init_grips=[left_grip, right_grip],
                names=['left_gripper', 'right_gripper'],
            )
        
        else:
            left_position, left_euler, left_grip = action[:3], action[3:6], action[6]
            right_position, right_euler, right_grip = action[7:10], action[10:13], action[13]
            positions = [left_position, right_position]
            eulers = [left_euler, right_euler]
            grips = [left_grip, right_grip]
            
            if not args.use_delta:
                converter.add_frame(positions, eulers, grips)
            else:
                converter.add_delta_frame(positions, eulers, grips)
    
    if converter is not None:
        converter.plot_trajectories_3d(names=['left_gripper', 'right_gripper'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pika trajectory data to absolute world coordinates.")
    parser.add_argument('--repo-id', 
                        type=str, 
                        required=True, 
                        help='The repository ID of the dataset.')
    parser.add_argument('--use-delta',
                        action='store_true', 
                        help='Use delta frame instead of absolute frame.')
    args = parser.parse_args()
    main(args)
