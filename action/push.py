import sapien.core as sapien
from sapien.core import Pose
from sapien.utils import Viewer
import numpy as np

from .core import Move_Tool
from perception import get_actor_by_name, get_pcd_from_actor
from transforms3d.euler import euler2quat
from utils import panda_x_direction_quant


class PandaPush:
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
        self.n_render_step=n_render_step

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
            direction:list,
            distance:float,        
    ):

        pose_pre_push, pose_post_push = self._compute_pose_for_push(object_name, direction, distance)   
        push_pose_path = self._interpolate_push_path(pose_pre_push, pose_post_push)
        
        self._open_gripper()

        if self._move_to_pose(push_pose_path[0], collision_avoid_all=True)==-1:
            print("Collision Avoidance Computation Fails.")
            if self._move_to_pose(push_pose_path[0])==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
            
            
        self._close_gripper()

        for push_pose in push_pose_path[1:]:
            if self._move_to_pose(push_pose)==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
        
        return 0


    def _compute_pose_for_push(
            self,
            object_name:str, 
            direction:list, 
            distance:float
    )-> Pose:
        object = get_actor_by_name(self.scene, object_name)
        obj_pcd = get_pcd_from_actor(object)
        obj_center = np.mean(obj_pcd, axis=0)
        long_axis = np.linalg.norm(np.max(obj_pcd, axis=0)[:2]-np.min(obj_pcd, axis=0)[:2])

        direction = direction/np.linalg.norm(direction)
        pose_pre_push, pose_post_push = sapien.Pose(), sapien.Pose()

        pose_pre_push.set_p(obj_center 
                            - np.array([direction[0], direction[1], 0]) * (long_axis + 0.02) 
                            + (0,0,-obj_center[2] +0.12 if obj_center[2]<0.12 else 0))
        pose_pre_push.set_q(panda_x_direction_quant(direction))

        pose_post_push.set_p(obj_center 
                             + np.array([direction[0], direction[1], 0]) * distance 
                             + (0,0,-obj_center[2] +0.12 if obj_center[2]<0.12 else 0))
        pose_post_push.set_q(panda_x_direction_quant(direction))

        return pose_pre_push, pose_post_push
    

    def _interpolate_push_path(self, pq1:Pose, pq2:Pose, step=0.1):
        p1, p2 = pq1.p, pq2.p
        distance = np.linalg.norm(p2 - p1)
        num_steps = int(distance / step)
        interpolated_p = np.array([p1 + (p2 - p1) * t / num_steps for t in range(num_steps + 1)])

        interpolated_pq=[]
        for p in interpolated_p:
            pq=Pose(p=p, q=pq1.q)
            interpolated_pq.append(pq)
        return interpolated_pq


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
        disturbation.set_p([disturbation.p[0], disturbation.p[1], disturbation.p[2]+0.001])
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