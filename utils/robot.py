import transforms3d.quaternions as quat
import numpy as np


# z forward, x up, y right
def panda_z_direction_quant(target_direction: list)->np.ndarray:
    x, y, z=target_direction[0], target_direction[1], target_direction[2]
    rotation_matrix = np.array([
        [1, y, x],
        [0, -x, y],
        [0, 0, z]
    ])

    return quat.mat2quat(rotation_matrix)


# z forward, x up, y right
def panda_xyz_direction_quant(
        target_x=[1,0,0],
        target_y=[0,1,0], 
        target_z=[0,0,1]
    )->np.ndarray:
    rotation_matrix = np.array([
        target_x,
        target_y,
        target_z
    ]).T

    return quat.mat2quat(rotation_matrix)


# x forward, z down, y right
def panda_x_direction_quant(target_direction: list)->np.ndarray:
    x, y, z=target_direction[0], target_direction[1], target_direction[2]
    rotation_matrix = np.array([
        [x, y, 0],
        [y, -x, 0],
        [z, 0, -1]
    ])

    return quat.mat2quat(rotation_matrix)