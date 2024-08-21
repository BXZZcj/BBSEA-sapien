import mplib.planner
import sapien.core as sapien
from sapien.core import Pose, Actor, ActorBuilder
from sapien.utils import Viewer
import mplib
import numpy as np
from PIL import Image
from transforms3d.affines import decompose44
from transforms3d.quaternions import mat2quat
from typing import Tuple

from perception import get_scene_pcd, get_pcd_from_obj
from scene.core import TaskScene
from scene.specified_object import Robot


class Move_Tool():
    def __init__(
        self,
        task_scene: TaskScene,
        robot: Robot,
    ):
        self.task_scene=task_scene
        self.robot=robot


    def setup_planner(
            self,
            collision_avoid_attach_obj="",
            collision_avoid_obj="",
            collision_avoid_all=False,
            collision_avoid_all_except=[],
            speed_factor:int=1,
    ) -> mplib.Planner:
        planner = mplib.Planner(
            urdf=self.robot.urdf_file_path,
            srdf=self.robot.srdf_file_path,
            user_link_names=self.robot.origin_link_names,
            user_joint_names=self.robot.origin_joint_names,
            move_group=self.robot.move_group,
            joint_vel_limits=self.robot.joint_vel_limits*speed_factor,
            joint_acc_limits=self.robot.joint_acc_limits*speed_factor,
        )

        def _updare_planner_attach_actor(
                planner:mplib.Planner, 
                actor:Actor,
                actor_builder:ActorBuilder
        ):
            for vis_record in actor_builder.get_visuals():
                if vis_record.type == "Box":
                    T, R, _, _ = decompose44(actor.get_pose().to_transformation_matrix() @ vis_record.pose.to_transformation_matrix())
                    planner.update_attached_box(
                        vis_record.scale.tolist(),
                        np.concatenate([T, mat2quat(R)]),
                    )
                elif vis_record.type=="Sphere":
                    T, R, _, _ = decompose44(actor.get_pose().to_transformation_matrix() @ vis_record.pose.to_transformation_matrix())
                    planner.update_attached_sphere(
                        vis_record.radius,
                        np.concatenate([T, mat2quat(R)]),
                    )
                elif vis_record.type=="Capsule":
                    r=vis_record.radius
                    l=vis_record.length
                    T, R, _, _ = decompose44(actor.get_pose().to_transformation_matrix() @ vis_record.pose.to_transformation_matrix())
                    planner.update_attached_box(
                        [r+l, r, r],
                        np.concatenate([T, mat2quat(R)]),
                    )
                else: # vis_record.type=="File":
                    T, R, _, _ = decompose44(actor.get_pose().to_transformation_matrix() @ vis_record.pose.to_transformation_matrix())
                    planner.update_attached_mesh(
                        vis_record.filename, 
                        np.concatenate([T, mat2quat(R)]),
                    )

        
        if collision_avoid_all or collision_avoid_all_except or collision_avoid_attach_obj:
            scene_pcd = get_scene_pcd(self.task_scene, collision_avoid_all_except+[collision_avoid_attach_obj])
            
            # from utils.visualization import visualize_pcd
            # visualize_pcd(scene_pcd)

            planner.update_point_cloud(scene_pcd)

            if collision_avoid_attach_obj:
                obj=self.task_scene.get_object_by_name(collision_avoid_attach_obj)
                _updare_planner_attach_actor(planner, obj, obj.get_builder())
            for mounted_obj in self.robot.mounted_obj:
                for builder in mounted_obj.get_articulation().get_builder().get_link_builders():
                    if builder.get_index()==mounted_obj.get_index():
                        actor_builder=builder
                        break
                _updare_planner_attach_actor(planner, mounted_obj, actor_builder)

        if collision_avoid_obj:
            pcd = get_pcd_from_obj(self.task_scene.get_object_by_name(collision_avoid_obj))
            planner.update_point_cloud(pcd)
        
        return planner
    

    # The IK method provided by mplib is unable to check collision avoidance
    def check_reachability_IK(
            self, 
            target_pose: Pose
    ) -> bool:
        planner = self.setup_planner(collision_avoid_all=True)
        target_pose=target_pose.p.tolist() + target_pose.q.tolist()
        init_qpos=self.robot.body.get_qpos().tolist()
        status, _ = planner.IK(target_pose, init_qpos)

        return status=="Success"
    
    
    def check_reachability_MP(
            self,
            target_pose: sapien.Pose,
            collision_avoid_attach_obj="",
            collision_avoid_obj="",
            collision_avoid_all=False,
            collision_avoid_all_except=[],
            mp_algo=[False, False, False, True],
    ) -> bool:
        screw_result_w_CA, screw_result_wo_CA, sampling_result_w_CA, sampling_result_wo_CA = \
        self.motion_planning(
            target_pose,
            collision_avoid_attach_obj,
            collision_avoid_obj,
            collision_avoid_all,
            collision_avoid_all_except,
            mp_algo,
        )

        return screw_result_w_CA['status'] == "Success", \
            screw_result_wo_CA['status'] == "Success", \
            sampling_result_w_CA['status'] == "Success", \
            sampling_result_wo_CA['status'] == "Success"
    

    def motion_planning(
            self,
            target_pose: sapien.Pose,
            collision_avoid_attach_obj="",
            collision_avoid_obj="",
            collision_avoid_all=False,
            collision_avoid_all_except=[],
            mp_algo=[False, False, False, True],
    ) -> Tuple[dict, dict, dict, dict]:
        """
        Parameters:
        ----------
        check_type : list, optional
            A list of boolean values indicating the type of motion planning algorithm to use:
            - [0]: Use screw motion planning with collision avoidance.
            - [1]: Use screw motion planning without collision avoidance.
            - [2]: Use sampling-based motion planning with collision avoidance.
            - [3]: Use sampling-based motion planning without collision avoidance.
        """
        planner=self.setup_planner(
            collision_avoid_attach_obj,
            collision_avoid_obj,
            collision_avoid_all,
            collision_avoid_all_except,
            )
        
        pose_list = list(target_pose.p)+list(target_pose.q)
        
        # CA stands for "collision avoidance"
        screw_result_w_CA = planner.plan_screw(pose_list, self.robot.body.get_qpos(), time_step=self.task_scene.time_step, 
                                    use_point_cloud=True, use_attach=True) \
                                        if mp_algo[0] else {"status":None}
        screw_result_wo_CA = planner.plan_screw(pose_list, self.robot.body.get_qpos(), time_step=self.task_scene.time_step, 
                                    use_point_cloud=False, use_attach=False) \
                                        if mp_algo[1] else {"status":None}
        sampling_result_w_CA = planner.plan_qpos_to_pose(pose_list, self.robot.body.get_qpos(), time_step=self.task_scene.time_step,
                                               use_point_cloud=True, use_attach=True) \
                                        if mp_algo[2] else {"status":None}
        sampling_result_wo_CA = planner.plan_qpos_to_pose(pose_list, self.robot.body.get_qpos(), time_step=self.task_scene.time_step,
                                               use_point_cloud=False, use_attach=False) \
                                        if mp_algo[3] else {"status":None}
        
        return screw_result_w_CA, screw_result_wo_CA, sampling_result_w_CA, sampling_result_wo_CA


    def follow_path(
            self, 
            mp_result: dict,
            n_render_step=4,
    ) -> None:
        active_joints = self.robot.body.get_active_joints()
        n_step = mp_result['position'].shape[0]
        n_driven_joints=mp_result['position'].shape[1]

        if n_step > 1000:
            raise Exception("The motion path is too long. That's usual, probably, the arm will tic, or make a circle.")

        for i in range(n_step):
            qf = self.robot.body.compute_passive_force()
            self.robot.body.set_qf(qf)
            for j in range(n_driven_joints):
                active_joints[j].set_drive_target(mp_result['position'][i][j])
                active_joints[j].set_drive_velocity_target(mp_result['velocity'][i][j])
            self.task_scene.step(render_step=i, n_render_step=n_render_step)

    
    def move_to_pose(    
            self,
            target_pose: sapien.Pose,
            collision_avoid_attach_obj="",
            collision_avoid_obj="",
            collision_avoid_all=False,
            collision_avoid_all_except=[],
            guarantee_screw_mp=False,
            speed_factor=1,
            n_render_step=4,
    ) -> int:
        planner=self.setup_planner(
            collision_avoid_attach_obj,
            collision_avoid_obj,
            collision_avoid_all,
            collision_avoid_all_except,
            speed_factor,
            )
        
        pose_list = list(target_pose.p)+list(target_pose.q)

        # Screw Algo
        mp_result = planner.plan_screw(pose_list, self.robot.body.get_qpos(), time_step=self.task_scene.time_step, 
                                    use_point_cloud=~guarantee_screw_mp, use_attach=~guarantee_screw_mp)
        if mp_result['status'] != "Success":
            # RTTConnect Algo
            mp_result = planner.plan_qpos_to_pose(pose_list, self.robot.body.get_qpos(), time_step=self.task_scene.time_step,
                                               use_point_cloud=True, use_attach=True)
            if mp_result['status'] != "Success":
                return -1
            
        self.follow_path(mp_result=mp_result, n_render_step=n_render_step)

        return 0