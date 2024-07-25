import mplib.planner
import sapien.core as sapien
from sapien.core import Pose
from sapien.utils import Viewer
import mplib
import numpy as np

from perception import get_pcd_from_actor, get_actor_by_name


class Move_Tool():
    def __init__(
        self,
        scene: sapien.Scene,
        viewer: Viewer,
        robot: sapien.Articulation, 
        urdf_file_path: str,
        srdf_file_path: str,
        move_group: str,
        joint_vel_limits=[],
        joint_acc_limits=[],
    ):
        self.scene=scene
        self.viewer=viewer
        self.robot=robot
        self.urdf_file_path=urdf_file_path
        self.srdf_file_path=srdf_file_path
        self.move_group=move_group
        self.joint_vel_limits=joint_vel_limits
        self.joint_acc_limits=joint_acc_limits


    def setup_planner(
            self,
            collision_avoid_attach_actor="",
            collision_avoid_actor="",
            collision_avoid_all=False,
            collision_avoid_all_except=[]
    ) -> mplib.Planner:
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.urdf_file_path,
            srdf=self.srdf_file_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.move_group,
            joint_vel_limits=self.joint_vel_limits,
            joint_acc_limits=self.joint_acc_limits)
        
        if collision_avoid_all or collision_avoid_all_except:
            combined_pcd=np.array([]).reshape(0, 3)
            for actor in self.scene.get_all_actors():
                if actor.get_name()!="ground" and actor.get_name() not in collision_avoid_all_except:
                    pcd = get_pcd_from_actor(actor)
                    actor_type = actor.get_builder().get_visuals()[0].type
                    combined_pcd = np.concatenate((combined_pcd, pcd), axis=0)
            planner.update_point_cloud(combined_pcd)
        if collision_avoid_actor:
            pcd = get_pcd_from_actor(get_actor_by_name(self.scene, collision_avoid_actor))
            planner.update_point_cloud(pcd)
        if collision_avoid_attach_actor:
            actor=get_actor_by_name(self.scene, collision_avoid_attach_actor)
            actor_type = actor.get_builder().get_visuals()[0].type
            if actor_type=="Box":
                planner.update_attached_box(
                    actor.get_builder().get_visuals()[0].scale.tolist(),
                    actor.get_pose().p.tolist()+actor.get_pose().q.tolist()
                    )
            elif actor_type=="Sphere":
                planner.update_attached_sphere(
                    actor.get_builder().get_visuals()[0].radius,
                    actor.get_pose().p.tolist()+actor.get_pose().q.tolist()
                    )
            elif actor_type=="Capsule":
                r=actor.get_builder().get_collisions()[0].radius
                l=actor.get_builder().get_collisions()[0].length
                planner.update_attached_box(
                    [r+l, r, r],
                    actor.get_pose().p.tolist()+actor.get_pose().q.tolist()
                    )
            else: # actor_type=="File":
                planner.update_attached_mesh(
                    actor.get_builder().get_collisions()[0].filename, 
                    actor.get_pose().p.tolist()+actor.get_pose().q.tolist()
                    )

        return planner
    

    # The IK method provided by mplib is unable to check collision avoidance
    def check_feasibility(
            self, 
            target_pose: Pose
    ) -> bool:
        planner = self.setup_planner(collision_avoid_all=True)
        target_pose=target_pose.p.tolist() + target_pose.q.tolist()
        init_qpos=self.robot.get_qpos().tolist()
        status, _ = planner.IK(target_pose, init_qpos)

        return status=="Success"

    
    def move_to_pose(    
            self,
            pose: sapien.Pose,
            collision_avoid_attach_actor="",
            collision_avoid_actor="",
            collision_avoid_all=False,
            collision_avoid_all_except=[],
            guarantee_screw_mp=False,
            n_render_step=4,
            time_step=1/100,
    ) -> int:
        
        def follow_path(result: dict) -> None:
            scene=self.scene
            active_joints = self.robot.get_active_joints()
            n_step = result['position'].shape[0]
            n_driven_joints=result['position'].shape[1]

            for i in range(n_step):  
                qf = self.robot.compute_passive_force(
                    gravity=True, 
                    coriolis_and_centrifugal=True)
                self.robot.set_qf(qf)
                for j in range(n_driven_joints):
                    active_joints[j].set_drive_target(result['position'][i][j])
                    active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
                scene.step()
                if i % n_render_step == 0:
                    scene.update_render()
                    self.viewer.render()
        

        planner=self.setup_planner(
            collision_avoid_attach_actor,
            collision_avoid_actor,
            collision_avoid_all,
            collision_avoid_all_except,
            )
        
        pose_list = list(pose.p)+list(pose.q)
        # Screw Algo
        result = planner.plan_screw(pose_list, self.robot.get_qpos(), time_step=time_step, 
                                    use_point_cloud=~guarantee_screw_mp, use_attach=~guarantee_screw_mp)
        if result['status'] != "Success":
            # RTTConnect Algo
            result = planner.plan_qpos_to_pose(pose_list, self.robot.get_qpos(), time_step=time_step, verbose=True,
                                               use_point_cloud=True, use_attach=True)
            if result['status'] != "Success":
                return -1
        follow_path(result=result)
        return 0