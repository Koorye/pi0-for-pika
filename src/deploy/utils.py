import math
import numpy as np


def delta_to_absolute_root_translation(delta, state, delta_with_grip=False):
    """
    Transform delta to absolute root translation,
    delta and state are in the same coordinate system.
    Params:
    - delta: [x, y, z, rx, ry, rz, grip] in world frame, x, y, z unit 0.001mm and rx, ry, rz unit 0.001 degree
    - state: [x, y, z, rx, ry, rz, grip] in world frame, x, y, z unit 0.001mm and rx, ry, rz unit 0.001 degree
    """
    pose6d = delta[:6] + state[:6]
    grip = delta[6]
    if delta_with_grip:
        grip += state[6]
    return np.concatenate([pose6d, [grip]])


def deg001_to_rad(deg001):
    """Convert 0.001 degrees to radians"""
    return deg001 * (math.pi / (180 * 1000))

def rad_to_deg001(rad):
    """Convert radians to 0.001 degrees"""
    return rad * (180 * 1000) / math.pi

def euler_to_rotation_matrix(rx, ry, rz):
    """Convert Euler angles (in radians) to rotation matrix (XYZ order)"""
    # XYZ order: first rx (roll), then ry (pitch), then rz (yaw)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    
    # Rotation matrix for XYZ order
    R = np.array([
        [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
        [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
        [-sy,   sx*cy,            cx*cy]
    ])
    return R

def delta_to_absolute_gripper_translation(delta, state, delta_with_grip=False):
    """
    Transform delta to absolute gripper translation,
    delta is in gripper frame and state is in world frame.
    
    Params:
    - delta: [x, y, z, rx, ry, rz, grip] in gripper frame (x=forward, z=up)
             units: x,y,z in 0.001mm, rx,ry,rz in 0.001 degree
    - state: [x, y, z, rx, ry, rz, grip] in world frame (z=up, x=forward)
             units: x,y,z in 0.001mm, rx,ry,rz in 0.001 degree
    
    Returns:
    - Absolute position and orientation in world frame [x, y, z, rx, ry, rz, grip]
    """
    # Extract state components
    state_pos = np.array(state[:3])  # [x, y, z]
    state_rot = np.array(state[3:6])  # [rx, ry, rz]
    state_grip = state[6]  # Gripper position
    
    # Extract delta components
    delta_pos = np.array(delta[:3])  # [x, y, z]
    delta_rot = np.array(delta[3:6])  # [rx, ry, rz]
    delta_grip = delta[6]  # Gripper position
    
    # Convert angles from 0.001 degrees to radians
    state_rot_rad = np.array([deg001_to_rad(a) for a in state_rot])
    delta_rot_rad = np.array([deg001_to_rad(a) for a in delta_rot])
    state_rot_rad[1] -= math.pi / 2  # Adjust for coordinate system difference
    
    # Get rotation matrix for state (world to gripper frame)
    R_state = euler_to_rotation_matrix(*state_rot_rad)
    
    # Transform delta position from gripper frame to world frame
    # Note: Rotation matrix transforms vectors from gripper frame to world frame
    delta_pos_world = R_state @ delta_pos
    
    # Calculate absolute position
    abs_pos = state_pos + delta_pos_world
    
    # Calculate absolute rotation
    # Get rotation matrix for delta (gripper frame)
    R_delta = euler_to_rotation_matrix(*delta_rot_rad)
    
    # Combined rotation: R_total = R_state * R_delta
    R_total = R_state @ R_delta
    
    # Convert combined rotation matrix back to Euler angles (XYZ order)
    # Calculate Euler angles from rotation matrix
    sy = math.sqrt(R_total[0,0]**2 + R_total[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        rx = math.atan2(R_total[2,1], R_total[2,2])
        ry = math.atan2(-R_total[2,0], sy)
        rz = math.atan2(R_total[1,0], R_total[0,0])
    else:
        rx = math.atan2(-R_total[1,2], R_total[1,1])
        ry = math.atan2(-R_total[2,0], sy)
        rz = 0
    
    # Convert back to 0.001 degrees
    ry += math.pi / 2  # Adjust for the coordinate system difference
    abs_rot = np.array([rx, ry, rz])
    abs_rot_deg001 = np.array([rad_to_deg001(a) for a in abs_rot])
    
    grip = delta_grip
    if delta_with_grip:
        grip += state_grip
    
    return np.concatenate([abs_pos, abs_rot_deg001, [grip]])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
                     #    x     y      z    rx     ry     rz   grip
    state = np.array([50000,    0, 50000,    0, 90000,     0, 60000])
    delta = np.array([100,    0,     0,    0, 10000, 10000, 60000])
    
    states = [state.copy()]
    for _ in range(100):
        state = delta_to_absolute_gripper_translation(delta, state)
        states.append(state.copy())
    
    # plot 3d x, y, z
    states = np.array(states)[:, :3] / 1000  # Convert to mm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Trajectory of End Effector')

    # x, y, z axis equal
    # ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for x, y, z axes
    ax.axis('equal')
    plt.show()
