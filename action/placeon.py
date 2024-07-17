import sapien.core as sapien
from sapien.core import Pose
from sapien.utils import Viewer
import numpy as np
from transforms3d import euler
from scipy.spatial import ConvexHull
from typing import Tuple

from .core import Move_Tool
from perception import get_pcd_from_actor, get_actor_by_name, dense_sample_pcd, get_pcd_normal
from transforms3d import euler, quaternions
from utils import panda_x_direction_quant, panda_z_direction_quant


class PandaPlaceOn:
    def __init__(
            self,
            viewer: Viewer,
            robot: sapien.Articulation, 
            urdf_file_path: str,
            srdf_file_path: str,
            gripper: str,
            time_step=1/100,
            joint_vel_limits=[],
            joint_acc_limits=[],
            n_render_step=4,
    ):
        self.viewer=viewer
        self.scene=self.viewer.scene
        self.robot=robot
        self.urdf_file_path=urdf_file_path
        self.srdf_file_path=srdf_file_path
        self.gripper=gripper
        self.time_step=time_step
        self.joint_vel_limits=joint_vel_limits
        self.joint_acc_limits=joint_acc_limits
        self.n_render_step = n_render_step

        self.move_tool = Move_Tool(   
            self.scene,
            self.viewer,
            self.robot, 
            self.urdf_file_path,
            self.srdf_file_path,
            self.gripper,
            self.joint_vel_limits,
            self.joint_acc_limits,
            )


    def PlaceOn(
            self,
            object_name:str,
    ):
        def _compute_pose_for_place(pcd: np.ndarray) -> Tuple[Pose, Pose]:
            pcd_normals = get_pcd_normal(pcd)
            filter_out_mask = pcd_normals[:, 2] < 0.95

            pcd_upward = pcd[~filter_out_mask]
            pcd_normals_upward = pcd_normals[~filter_out_mask]
            
            # TODO sampling can be biased towards points that belong to larger surface areas
            num_candidates = (~filter_out_mask).sum()
            if num_candidates == 0:
                return None, None
            
            attempt_times = 100
            for i in range(attempt_times):
                candidate_indices = np.random.choice(
                    num_candidates,
                    size=min(1000, num_candidates),
                    replace=True,
                )

                # compute gripper orientation
                place_z_direction = -1 * pcd_normals_upward[candidate_indices].copy().mean(axis=0)
                place_z_direction /= np.linalg.norm(place_z_direction)
                place_z_quant = panda_z_direction_quant(place_z_direction)

                place_x_angle = np.random.uniform(np.pi, -np.pi)
                place_x_direction = (np.cos(place_x_angle), np.sin(place_x_angle), 0)
                place_x_quant = panda_x_direction_quant(place_x_direction)

                place_direction_quant = place_x_quant * place_z_quant

                # computer gripper position
                contact_position = pcd_upward[candidate_indices].copy().mean(axis=0)

                grasped_obj_pcd = get_pcd_from_actor(get_actor_by_name(self.grasped_obj))
                pcd_projections = np.dot(grasped_obj_pcd, place_z_direction)
                span = np.max(pcd_projections) - np.min(pcd_projections)
                gripper_deep = 0.05
                place_distance = span-gripper_deep

                place_pose = Pose(
                    p = contact_position + (0.12+place_distance) * (-place_z_direction),
                    q = place_direction_quant
                )

                preplace_pose = Pose(
                    p = place_pose.p + np.random.uniform(0.1, 0.2) * (-place_z_direction),
                    q = place_direction_quant
                )

                return place_pose, preplace_pose


        object_pcd = dense_sample_pcd(get_pcd_from_actor(get_actor_by_name(self.scene, object_name)))
        place_pose, preplace_pose = _compute_pose_for_place(object_pcd)

        if self._move_to_pose(preplace_pose, collision_avoid_all=True)==-1:
            print("Collision Avoidance Computation Fails.")
            if self._move_to_pose(preplace_pose)==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
        
        if self._move_to_pose(place_pose)==-1:
            print("Inverse Kinematics Computation Fails.")
            return -1
        
        self._open_gripper()

        self.grasped_obj=None
        
        if self._move_to_pose(preplace_pose)==-1:
            print("Inverse Kinematics Computation Fails.")
            return -1
        
        return 0