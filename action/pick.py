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
from utils import panda_x_direction_quant


class PandaPick:
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


    def execute(
            self,
            object_name:str,
    ):
        self._open_gripper()

        # object_pcd = dense_sample_pcd(get_pcd_from_actor(get_actor_by_name(self.scene, object_name)))
        # grasp_pose, pregrasp_pose = self._compute_pose_for_pick(object_pcd)

        # if self._move_to_pose(pregrasp_pose, collision_avoid_all=True)==-1:
        #     print("Collision Avoidance Computation Fails.")
        #     if self._move_to_pose(pregrasp_pose)==-1:
        #         print("Inverse Kinematics Computation Fails.")
        #         return -1

        # if self._move_to_pose(grasp_pose)==-1:
        #     print("Inverse Kinematics Computation Fails.")
        #     return -1
        
        # self._close_gripper()
        
        # if self._move_to_pose(pregrasp_pose, collision_avoid_attach_actor=object_name)==-1:
        #     print("Inverse Kinematics Computation Fails.")
        #     return -1
        
        return 0
    

    def _compute_pose_for_pick(self, pcd: np.ndarray, pushin_more=True)-> Tuple[Pose, Pose]:
        pcd_normals = get_pcd_normal(pcd)
        
        pointing_down = pcd_normals[:, 2] < 0.0
        P = np.ones(shape=len(pcd), dtype=np.float64)
        if not pointing_down.all():
            # sample the ones pointing up more often
            P = np.exp(pcd_normals[:, 2] * 1e2)
        P /= P.sum()

        attempt_count = 0
        attempt_times = 100
        for i in range(attempt_times):
            candidate_indices = np.random.choice(
                len(pcd),
                size=min(1000, len(pcd)),
                p=P,
                replace=True,
            )
            candidate_points = pcd[candidate_indices].copy()
            position = candidate_points.mean(axis=0)
            position[2] = pcd[:, 2].mean()

            panda_gripper_q = self._cal_pick_orientation(candidate_points)

            if pushin_more:
                pushin_distance = (len(candidate_indices) - attempt_count) / len(candidate_indices) * (0.04 - 0.03) + 0.03
            else:
                pushin_distance = attempt_count / len(candidate_indices) * (0.04 - 0.03) + 0.03

            # randomlization
            normal = pcd_normals[candidate_indices[0]]
            if normal[2] < 0:
                normal *= -1.0
            grasp_pose = Pose(
                p = position - normal * pushin_distance + [0,0,0.12],
                q = panda_gripper_q,
            )

            pregrasp_pose = Pose(
                p = position + normal * ((pcd[:,2].max() - pcd[:,2].mean()) + np.random.uniform(0.1, 0.2)),
                q = panda_gripper_q,
            )
            
            if self.move_tool.check_feasibility(grasp_pose) \
                and self.move_tool.check_feasibility(pregrasp_pose):
                break
            else:
                attempt_count+=1
                grasp_pose, pregrasp_pose = None, None
                continue

        return grasp_pose, pregrasp_pose      

    
    def _cal_pick_orientation(self, pcd: np.ndarray) -> np.array:
        # Project all points to the XY plane (top-down view)
        xy_points = pcd[:, :2]
    
        # Calculate the center of the points
        center = np.mean(xy_points, axis=0)

        # Calculate the convex hull of the points
        hull = ConvexHull(xy_points)

        # For each edge, compute the point on this edge that's closest to the centroid
        edges = [hull.points[simplex] for simplex in hull.simplices]
        closest_points_to_center = []
        for edge in edges:
            vector = edge[1] - edge[0]
            t = np.dot(center - edge[0], vector) / np.dot(vector, vector)
            t = np.clip(t, 0, 1)
            closest_point = edge[0] + t * vector
            closest_points_to_center.append(closest_point)
    
        # Select the edge for which the closest point is also closest to the centroid
        closest_point_index = np.argmin([np.linalg.norm(point - center) for point in closest_points_to_center])
        direction_vector = closest_points_to_center[closest_point_index] - center + [0]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Because the panda gripper is symmetric along the x-axis
        # So we need to select a best x orientation of panda gripper, which could to the most decrease the movement of gripper.
        panda_x_direction_mirror1 = np.cross([0,0,1], np.concatenate((direction_vector, np.array([0]))))
        panda_x_direction_mirror2 = np.cross(np.concatenate((direction_vector, np.array([0]))), [0,0,1])
        panda_q_mirror1 = panda_x_direction_quant(panda_x_direction_mirror1)
        panda_q_mirror2 = panda_x_direction_quant(panda_x_direction_mirror2)

        panda_gripper_q_cur = self.robot.get_links()[-3].get_pose().q.tolist()

        final_gripper_q = panda_q_mirror1 if sum(panda_gripper_q_cur*panda_q_mirror1) < sum(panda_gripper_q_cur*panda_q_mirror2) else panda_q_mirror2
        
        return final_gripper_q


    def _move_to_pose(
            self,
            pose: sapien.Pose,
            collision_avoid_attach_actor="",
            collision_avoid_actor="",
            collision_avoid_all=False,
    ) -> int:
        
        return self.move_tool.move_to_pose(
            pose,
            collision_avoid_attach_actor,
            collision_avoid_actor,
            collision_avoid_all,
            self.n_render_step,
            self.time_step,
            )


    def _open_gripper(
            self, 
            target=0.4
    ) -> None:
        # This disturbation aims to make the gripper open process more stable.
        disturbation=self.robot.get_links()[-3].get_pose()
        disturbation.set_p([disturbation.p[0]+0.1, disturbation.p[1]+0.1, disturbation.p[2]+0.001])
        self._move_to_pose(disturbation)

        for i in range(100): 
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for joint in self.robot.get_active_joints()[-2:]:
                joint.set_drive_target(target)   
            self.robot.set_qpos(self.robot.get_qpos())            
            self.scene.step()
            if i % self.n_render_step == 0:
                self.scene.update_render()
                self.viewer.render()


    def _close_gripper(self) -> None:
        # This disturbation aims to make the gripper close process more stable.
        disturbation=self.robot.get_links()[-3].get_pose()
        disturbation.set_p([disturbation.p[0], disturbation.p[1], disturbation.p[2]+0.001])
        self._move_to_pose(disturbation)

        for joint in self.robot.get_active_joints()[-2:]:
            joint.set_drive_target(0)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % self.n_render_step == 0:
                self.scene.update_render()
                self.viewer.render()