import mplib.planner
import sapien.core as sapien
from sapien.core import Pose, Actor, ActorBuilder
from sapien.utils import Viewer
import mplib
import numpy as np
from PIL import Image
from transforms3d.affines import decompose44
from transforms3d.quaternions import mat2quat

from perception import get_pcd_from_actor, get_actor_by_name, get_pcd_from_obj
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
        
        if collision_avoid_all or collision_avoid_all_except:
            combined_pcd=np.array([]).reshape(0, 3)
            for obj in self.task_scene.object_list:
                if obj.get_name() not in collision_avoid_all_except:
                    pcd = get_pcd_from_obj(obj, dense_sample_convex=True)
                    combined_pcd = np.concatenate((combined_pcd, pcd), axis=0)
            
            import open3d as o3d
            pcd_ = o3d.geometry.PointCloud()
            pcd_.points = o3d.utility.Vector3dVector(combined_pcd)
            o3d.visualization.draw_geometries([pcd_], window_name="Open3D Point Cloud Visualization")

            planner.update_point_cloud(combined_pcd)
        if collision_avoid_obj:
            pcd = get_pcd_from_obj(self.task_scene.get_object_by_name(collision_avoid_obj), dense_sample_convex=True)
            planner.update_point_cloud(pcd)
        if collision_avoid_attach_obj:
            obj=self.task_scene.get_object_by_name(collision_avoid_attach_obj)
            _updare_planner_attach_actor(planner, obj, obj.get_builder())
        
        for mounted_obj in self.robot.mounted_obj:
            for builder in mounted_obj.get_articulation().get_builder().get_link_builders():
                if builder.get_name()==mounted_obj.get_name():
                    actor_builder=builder
                    break
            _updare_planner_attach_actor(planner, mounted_obj, actor_builder)

        return planner
    

    # The IK method provided by mplib is unable to check collision avoidance
    def check_feasibility(
            self, 
            target_pose: Pose
    ) -> bool:
        planner = self.setup_planner(collision_avoid_all=True)
        target_pose=target_pose.p.tolist() + target_pose.q.tolist()
        init_qpos=self.robot.robot_articulation.get_qpos().tolist()
        status, _ = planner.IK(target_pose, init_qpos)

        return status=="Success"

    
    def move_to_pose(    
            self,
            pose: sapien.Pose,
            collision_avoid_attach_obj="",
            collision_avoid_obj="",
            collision_avoid_all=False,
            collision_avoid_all_except=[],
            guarantee_screw_mp=False,
            speed_factor=1,
            n_render_step=4,
    ) -> int:
        
        def follow_path(result: dict) -> None:
            active_joints = self.robot.robot_articulation.get_active_joints()
            n_step = result['position'].shape[0]
            n_driven_joints=result['position'].shape[1]

            for i in range(n_step):                     
                qf = self.robot.robot_articulation.compute_passive_force(
                    gravity=True, 
                    coriolis_and_centrifugal=True,
                    # external=False,
                )
                self.robot.robot_articulation.set_qf(qf)
                for j in range(n_driven_joints):
                    active_joints[j].set_drive_target(result['position'][i][j])
                    active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
                self.task_scene.step(render_step=i, n_render_step=n_render_step)
        

        planner=self.setup_planner(
            collision_avoid_attach_obj,
            collision_avoid_obj,
            collision_avoid_all,
            collision_avoid_all_except,
            speed_factor,
            )
        
        pose_list = list(pose.p)+list(pose.q)
        # Screw Algo
        result = planner.plan_screw(pose_list, self.robot.robot_articulation.get_qpos(), time_step=self.task_scene.time_step, 
                                    use_point_cloud=~guarantee_screw_mp, use_attach=~guarantee_screw_mp)
        if result['status'] != "Success":
            # RTTConnect Algo
            result = planner.plan_qpos_to_pose(pose_list, self.robot.robot_articulation.get_qpos(), time_step=self.task_scene.time_step,
                                               use_point_cloud=True, use_attach=True)
            if result['status'] != "Success":
                return -1
        follow_path(result=result)
        return 0