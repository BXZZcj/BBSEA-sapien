import sapien.core as sapien
from sapien.core import Pose, Actor, Articulation
from sapien.utils import Viewer
import numpy as np
from transforms3d import euler, axangles
from scipy.spatial import ConvexHull
from typing import Tuple, Union
import quaternion

from .core import Move_Tool
from perception import get_pcd_from_actor, \
    get_pcd_from_obj, \
    get_actor_by_name, \
    get_normals_from_actor, \
    get_pcd_normals_from_obj
from transforms3d import euler, quaternions
from utils import panda_x_direction_quant, panda_z_direction_quant, panda_xyz_direction_quant
from scene.core import TaskScene, SpecifiedObject
from scene.specified_object import StorageFurniture, Robot

# the distance between the move group center with the grippers center
DMG2G = 0.1
# the figger length of the gripper
FL = 0.048


class PandaPrimitives:
    def __init__(
            self,
            task_scene: TaskScene,
            robot: Robot,
            n_render_step=4,
    ):
        self.task_scene = task_scene
        self.robot=robot
        self.n_render_step=n_render_step

        self.move_tool = Move_Tool(   
            self.task_scene,
            self.robot,
            )    
        
        self.grasped_obj = None


    def _move_to_pose(
            self,
            gripper_pose: sapien.Pose,
            collision_avoid_attach_obj="",
            collision_avoid_obj="",
            collision_avoid_all=False,
            collision_avoid_all_except=[],
            speed_factor=1,
            guarantee_screw_mp=False,
            distinguish_gripper_movegroup=True,
    ) -> int:
        if distinguish_gripper_movegroup:
            gripper_direction = (euler.quat2mat(gripper_pose.q) @ np.array([0,0,1]).T).T
            gripper_direction = gripper_direction/np.linalg.norm(gripper_direction)

            move_group_pose = sapien.Pose()
            move_group_pose.set_p(gripper_pose.p - DMG2G*gripper_direction)
            move_group_pose.set_q(gripper_pose.q)
        else:
            move_group_pose = gripper_pose

        return self.move_tool.move_to_pose(
            move_group_pose,
            collision_avoid_attach_obj,
            collision_avoid_obj,
            collision_avoid_all,
            collision_avoid_all_except,
            guarantee_screw_mp,
            speed_factor,
            self.n_render_step,
            )
    

    def _open_gripper(
            self, 
            target=0.4
    ) -> None:
        # This disturbation aims to make the gripper open process more stable.
        for link in self.robot.robot_articulation.get_links():
            if link.get_name()==self.robot.move_group:
                disturbation=link.get_pose()
        disturbation.set_p([disturbation.p[0], disturbation.p[1], disturbation.p[2]+0.001])
        self._move_to_pose(disturbation, distinguish_gripper_movegroup=False) 
        

        for joint in self.robot.robot_articulation.get_active_joints()[-2:]:
            joint.set_drive_target(target)   
        last_gripper_qpos=None
        for i in range(100): 
            qf = self.robot.robot_articulation.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True,
                # external=False,
            )
            self.robot.robot_articulation.set_qf(qf)  

            self.task_scene.step(render_step=i, n_render_step=self.n_render_step)

            # In case the range(100) is too long
            current_gripper_qpos = self.robot.robot_articulation.get_qpos()[-2:]
            if np.array_equal(current_gripper_qpos, last_gripper_qpos):
                break
            else:
                last_gripper_qpos=current_gripper_qpos

        # We assume when the gripper opens, the grasped object will fall
        self.grasped_obj = None


    def _close_gripper(self) -> None:
        # This disturbation aims to make the gripper close process more stable.
        for link in self.robot.robot_articulation.get_links():
            if link.get_name()==self.robot.move_group:
                disturbation=link.get_pose()
        disturbation.set_p([disturbation.p[0], disturbation.p[1], disturbation.p[2]+0.001])
        self._move_to_pose(disturbation, distinguish_gripper_movegroup=False)

        for joint in self.robot.robot_articulation.get_active_joints()[-2:]:
            # joint.set_friction(10)
            # joint.set_drive_property(stiffness=100, damping=100, force_limit=10)
            joint.set_drive_target(0)
        last_gripper_qpos=None
        for i in range(150):  
            qf = self.robot.robot_articulation.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
                # external=False,
            )
            self.robot.robot_articulation.set_qf(qf)   

            self.task_scene.step(render_step=i, n_render_step=self.n_render_step)

            # In case the range(100) is too long
            current_gripper_qpos = self.robot.robot_articulation.get_qpos()[-2:]
            if np.array_equal(current_gripper_qpos, last_gripper_qpos):
                break
            else:
                last_gripper_qpos=current_gripper_qpos


    def Push(
            self,
            object_name:str,
            direction:list,
            distance:float,
    ):
        assert len(direction)==2 or len(direction)==3, "The direction shape is invalid." 
        if len(direction)==2:
            direction=np.concatenate((direction,[0]))
        direction = direction/np.linalg.norm(direction)
        
        def _compute_pose_for_push(
                object_name:str, 
                direction:list, 
                distance:float
        )-> Pose:
            obj = self.task_scene.get_object_by_name(object_name)
            obj_pcd = get_pcd_from_obj(obj)
            obj_center = np.mean(obj_pcd, axis=0)

            projections = np.dot(obj_pcd, direction)
            long_axis = projections.max() - projections.min()
            height=obj_pcd[:,2].max()-obj_pcd[:,2].min()

            # Make the gripper pushing plane parellel with the object pushed plane
            # Especially when there is a camera mounted on the end effector
            rot_ax = np.cross([0,0,1], direction)
            rot_mx = axangles.axangle2mat(rot_ax, np.deg2rad(-55))
            gripper_x_direction = rot_mx @ direction
            gripper_x_direction = gripper_x_direction / np.linalg.norm(gripper_x_direction)

            pose_pre_push, pose_post_push = sapien.Pose(), sapien.Pose()
            # The z value should be a little bigger than 0 (z value of the table top), 
            # or the collision avoidance equation will never be solved,
            # because the gripper will always contact with the table top  
            pose_pre_push.set_p(np.array([obj_center[0], obj_center[1], obj_pcd[:,2].min()+height/3]) - direction * (long_axis/2+0.1))
            pose_pre_push.set_q(panda_x_direction_quant(gripper_x_direction))

            pose_post_push.set_p(np.array([obj_center[0], obj_center[1], obj_pcd[:,2].min()+height/3]) + direction * distance)
            pose_post_push.set_q(panda_x_direction_quant(gripper_x_direction))

            return pose_pre_push, pose_post_push
        

        def _interpolate_push_path(
                pq1:Pose, 
                pq2:Pose, 
                step=np.inf,
        ) -> int:
            p1, p2 = pq1.p, pq2.p
            distance = np.linalg.norm(p2 - p1)
            num_steps = max(int(distance / step), 1)
            interpolated_p = np.array([p1 + (p2 - p1) * t / num_steps for t in range(num_steps + 1)])

            interpolated_pq=[]
            for p in interpolated_p:
                pq=Pose(p=p, q=pq1.q)
                interpolated_pq.append(pq)
            return interpolated_pq

        pose_pre_push, pose_post_push = _compute_pose_for_push(object_name, direction, distance)   
        push_pose_path = _interpolate_push_path(pose_pre_push, pose_post_push)
        
        self._open_gripper()

        if self._move_to_pose(push_pose_path[0], collision_avoid_all=True)==-1:
            raise Exception()
            print("Collision Avoidance Computation Fails.")
            if self._move_to_pose(push_pose_path[0])==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
            
        self._close_gripper()

        for push_pose in push_pose_path[1:]:
            if self._move_to_pose(push_pose, guarantee_screw_mp=True, speed_factor=0.1)==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
        
        return 0
    

    def Pick(
            self,
            object_name:str,
    ):
        def _compute_pose_for_pick(
                obj: Union[Actor,Articulation,SpecifiedObject], 
                pushin_more=True
        )-> Tuple[Pose, Pose]:
            pcd, pcd_normals = get_pcd_normals_from_obj(obj)
            
            pointing_down = pcd_normals[:, 2] < 0.0
            P = np.ones(shape=len(pcd), dtype=np.float64)
            if not pointing_down.all():
                # sample the ones pointing up more often
                P = np.exp((pcd_normals[:, 2] * 1e2).astype(np.float64))
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
                position[2] = max(pcd[:, 2].mean(), pcd[:,2].max()-FL)

                panda_gripper_q = _cal_pick_orientation(candidate_points)

                if pushin_more:
                    pushin_distance = (len(candidate_indices) - attempt_count) / len(candidate_indices) * (0.04 - 0.03) + 0.03
                else:
                    pushin_distance = attempt_count / len(candidate_indices) * (0.04 - 0.03) + 0.03

                # randomlization
                normal = pcd_normals[candidate_indices[0]]
                if normal[2] < 0:
                    normal *= -1.0
                grasp_pose = Pose(
                    # p = position - normal * pushin_distance + [0,0,0.12],
                    p = position,
                    q = panda_gripper_q,
                )

                pregrasp_pose = Pose(
                    p = position + normal * (pcd[:,2].max() - pcd[:,2].mean() + np.random.uniform(0.2, 0.3)),
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

        
        def _cal_pick_orientation(pcd: np.ndarray) -> np.array:
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

            panda_gripper_q_cur = self.robot.robot_articulation.get_links()[-3].get_pose().q.tolist()

            final_gripper_q = panda_q_mirror1 if sum(panda_gripper_q_cur*panda_q_mirror1) < sum(panda_gripper_q_cur*panda_q_mirror2) else panda_q_mirror2
            
            return final_gripper_q
        

        self._open_gripper()

        grasp_pose, pregrasp_pose = _compute_pose_for_pick(self.task_scene.get_object_by_name(object_name))
        if grasp_pose==None or pregrasp_pose==None:
            return -1

        if self._move_to_pose(pregrasp_pose, collision_avoid_all=True)==-1:
            print("Collision Avoidance Computation Fails.")
            raise Exception()
            if self._move_to_pose(pregrasp_pose)==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1

        if self._move_to_pose(grasp_pose)==-1:
            print("Inverse Kinematics Computation Fails.")
            return -1
        
        self._close_gripper()
        
        if self._move_to_pose(pregrasp_pose, collision_avoid_attach_obj=object_name)==-1:
            print("Inverse Kinematics Computation Fails.")
            return -1
        
        # TODO Judge whether the target object has been grasped successfully.
        self.grasped_obj = object_name
        
        return 0
    

    def PlaceOn(
            self,
            object_name:str,
    ):
        def _compute_pose_for_place(obj: Union[Actor,Articulation,SpecifiedObject]) -> Tuple[Pose, Pose]:
            pcd, pcd_normals = get_pcd_normals_from_obj(obj)
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

                place_x_angle = np.random.uniform(-np.pi, np.pi)
                place_x_quant = euler.euler2quat(0,0, place_x_angle)

                place_direction_quant = quaternions.qmult(place_z_quant,place_x_quant)

                # computer gripper position
                contact_position = pcd_upward[candidate_indices].copy().mean(axis=0)

                grasped_obj_pcd = get_pcd_from_actor(get_actor_by_name(self.task_scene.scene, self.grasped_obj))
                pcd_projections = np.dot(grasped_obj_pcd, place_z_direction)
                span = np.max(pcd_projections) - np.min(pcd_projections)
                gripper_deep = 0.05
                place_distance = max(span-gripper_deep, 0)

                place_pose = Pose(
                    p = contact_position + place_distance * -place_z_direction,
                    q = place_direction_quant
                )

                preplace_pose = Pose(
                    p = place_pose.p + np.random.uniform(0.1, 0.2) * (-place_z_direction),
                    q = place_direction_quant
                )
                
                if self.move_tool.check_feasibility(place_pose) \
                    and self.move_tool.check_feasibility(preplace_pose):
                    break
                else:
                    place_pose, preplace_pose = None, None
                    continue

            return place_pose, preplace_pose


        place_pose, preplace_pose = _compute_pose_for_place(self.task_scene.get_object_by_name(object_name))
        if place_pose==None or preplace_pose==None:
            return -1

        if self._move_to_pose(preplace_pose, collision_avoid_attach_obj=self.grasped_obj)==-1:
            print("Collision Avoidance Computation Fails.")
            raise Exception()
            if self._move_to_pose(preplace_pose)==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
        
        if self._move_to_pose(place_pose, collision_avoid_attach_obj=self.grasped_obj)==-1:
            print("Collision Avoidance Computation Fails.")
            if self._move_to_pose(place_pose)==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
        
        self._open_gripper()

        self.grasped_obj=None
        
        if self._move_to_pose(preplace_pose, collision_avoid_all=True)==-1:
            print("Collision Avoidance Computation Fails.")
            if self._move_to_pose(preplace_pose)==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
        
        return 0
    

    def PlaceAt(
            self,
            place_pos:str,
    ):
        def _compute_pose_for_place(place_pos):
            place_pose = Pose(
                p = place_pos,
                q = self.robot.robot_articulation.get_links()[-3].get_pose().q
            )

            preplace_pose = Pose(
                p = place_pos + np.random.uniform(0.2, 0.3) * np.array([0,0,1]),
                q = self.robot.robot_articulation.get_links()[-3].get_pose().q
            )

            return place_pose, preplace_pose 

        
        place_pose, preplace_pose = _compute_pose_for_place(place_pos)
        if place_pose==None or preplace_pose==None:
            return -1

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
    

    def DrawerOpen(self, handle_name: str, target_open_degree:float=1):
        assert target_open_degree>=0 and target_open_degree<=1, "The target open degree should be between 0 and 1."
        
        handle_parent_name = handle_name.split("_")[0]
        handle_parent : StorageFurniture = self.task_scene.get_object_by_name(handle_parent_name)
        current_open_degree = handle_parent.get_open_degree_by_name(handle_name)
        if current_open_degree>=target_open_degree:
            return 0

        def _compute_pose_for_grasp(handle_name: str, target_open_degree:float):
            handle_parent_name = handle_name.split("_")[0]
            handle_parent : StorageFurniture = self.task_scene.get_object_by_name(handle_parent_name)
            
            handle_pcd = handle_parent.get_handle_pcd_by_name(handle_name)
            parent_pcd = handle_parent.get_pcd()

            handle_center = handle_pcd.mean(axis=0)
            parent_center = parent_pcd.mean(axis=0)

            pull_direction = handle_center[:2] - parent_center[:2]
            # We assume the pull direction would be parellel with the horizontal ground
            pull_direction = np.concatenate((pull_direction, [0]))
            pull_direction = pull_direction / np.linalg.norm(pull_direction)

            open_limit = handle_parent.get_open_limit_by_name(handle_name)
            target_open_distance = target_open_degree * open_limit
            current_open_distance = handle_parent.get_open_distance_by_name(handle_name)
            open_displacement = target_open_distance - current_open_distance

            handle_direction = np.cross(pull_direction, np.array([0,0,1]))
            handle_direction = handle_direction/np.linalg.norm(handle_direction)
            _tmp_projection = np.dot(handle_pcd, handle_direction)
            handle_half_length = (_tmp_projection.max() - _tmp_projection.min())/2

            _tmp_projection = np.dot(handle_pcd, pull_direction)
            handle_radius = (_tmp_projection.max() - _tmp_projection.min())/2

            sample_count = 1000
            p_bias_candidate = np.linspace(-handle_half_length, handle_half_length, sample_count)
            P=5-5*np.abs(np.linspace(-1,1, sample_count))
            P=P/P.sum()
            p_bias = p_bias_candidate[np.random.choice(sample_count, size=1, p=P).item()]
            
            grasp_pose=sapien.Pose()
            grasp_pose.set_p(handle_parent.get_handle_pcd_by_name(handle_name).mean(axis=0) + 
                             p_bias * handle_direction - 
                             2*handle_radius * pull_direction)

            sample_count = 1000
            circle_curve = lambda x : (1-(x-1)**2)**0.5
            P=-1e10*np.log(circle_curve(np.abs(np.linspace(-1,1, sample_count))))
            P=P/P.sum()
            q_x_bias_candidate = np.linspace(-np.pi/6, np.pi/6, sample_count)
            q_y_bias_candidate = np.linspace(-np.pi/6, np.pi/6, sample_count)
            q_z_bias_candidate = np.linspace(-np.pi/6, np.pi/6, sample_count)
            q_x_bias = q_x_bias_candidate[np.random.choice(sample_count, size=1, p=P).item()]
            q_y_bias = q_y_bias_candidate[np.random.choice(sample_count, size=1, p=P).item()]
            q_z_bias = q_z_bias_candidate[np.random.choice(sample_count, size=1, p=P).item()]
            
            base_z_direction = -pull_direction
            base_z_direction=base_z_direction/np.linalg.norm(base_z_direction)
            base_x_direction = np.cross([0,0,1], base_z_direction)
            base_x_direction=base_x_direction/np.linalg.norm(base_x_direction)
            base_y_direction = np.cross(base_z_direction, base_x_direction)
            base_y_direction=base_y_direction/np.linalg.norm(base_y_direction)

            R = euler.euler2mat(q_z_bias, q_y_bias, q_x_bias, axes='rzyx')

            grasp_x_direction = (R@base_x_direction.T).T
            grasp_y_direction = (R@base_y_direction.T).T
            grasp_z_direction = (R@base_z_direction.T).T
            grasp_pose.set_q(panda_xyz_direction_quant(grasp_x_direction, grasp_y_direction, grasp_z_direction))

            ungrasp_pose=sapien.Pose()
            ungrasp_pose.set_p(grasp_pose.p+pull_direction*open_displacement)
            ungrasp_pose.set_q(grasp_pose.q)

            pregrasp_pose=sapien.Pose()
            pregrasp_pose.set_p(grasp_pose.p+
                                np.random.uniform(max(0.05, open_displacement/3), open_displacement)*pull_direction + 
                                np.random.randn(3)/50)
            pregrasp_pose.set_q(euler.euler2quat(*(euler.quat2euler(grasp_pose.q)+np.random.randn(3)/50)))

            return pregrasp_pose, grasp_pose, ungrasp_pose

        assert "handle" in handle_name, "Only the handle of target object is graspable."

        self._open_gripper()

        pregrasp_pose, grasp_pose, ungrasp_pose = _compute_pose_for_grasp(handle_name, target_open_degree)
        if grasp_pose==None or pregrasp_pose==None or ungrasp_pose==None:
            return -1

        if self._move_to_pose(pregrasp_pose, collision_avoid_all=True)==-1:
            raise Exception()
            print("Collision Avoidance Computation Fails.")
            if self._move_to_pose(pregrasp_pose)==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
        
        if self._move_to_pose(grasp_pose)==-1:
            print("Inverse Kinematics Computation Fails.")
            return -1

        self._close_gripper()
        
        if self._move_to_pose(ungrasp_pose, speed_factor=0.3, guarantee_screw_mp=True)==-1:
            print("Inverse Kinematics Computation Fails.")
            return -1
        
        self._open_gripper()
        
        return 0
    

    def DrawerClose(self, handle_name: str, target_open_degree:float=0):
        assert target_open_degree>=0 and target_open_degree<=1, "The target open degree should be between 0 and 1."
        
        handle_parent_name = handle_name.split("_")[0]
        handle_parent : StorageFurniture = self.task_scene.get_object_by_name(handle_parent_name)
        current_open_degree = handle_parent.get_open_degree_by_name(handle_name)
        if current_open_degree<=target_open_degree:
            return 0

        def _compute_pose_for_grasp(handle_name: str, target_open_degree:float):
            handle_parent_name = handle_name.split("_")[0]
            handle_parent : StorageFurniture = self.task_scene.get_object_by_name(handle_parent_name)
            
            handle_pcd = handle_parent.get_handle_pcd_by_name(handle_name)
            parent_pcd = handle_parent.get_pcd()

            handle_center = handle_pcd.mean(axis=0)
            parent_center = parent_pcd.mean(axis=0)

            push_direction = parent_center[:2] - handle_center[:2] 
            # We assume the pull direction would be parellel with the horizontal ground
            push_direction = np.concatenate((push_direction, [0]))
            push_direction = push_direction / np.linalg.norm(push_direction)

            open_limit = handle_parent.get_open_limit_by_name(handle_name)
            target_open_distance = target_open_degree * open_limit
            current_open_distance = handle_parent.get_open_distance_by_name(handle_name)
            close_displacement = current_open_distance - target_open_distance 

            handle_direction = np.cross(push_direction, np.array([0,0,1]))
            handle_direction = handle_direction/np.linalg.norm(handle_direction)
            _tmp_projection = np.dot(handle_pcd, handle_direction)
            handle_half_length = (_tmp_projection.max() - _tmp_projection.min())/2

            _tmp_projection = np.dot(handle_pcd, push_direction)
            handle_radius = (_tmp_projection.max() - _tmp_projection.min())/2

            sample_count = 1000
            p_bias_candidate = np.linspace(-handle_half_length, handle_half_length, sample_count)
            P=5-5*np.abs(np.linspace(-1,1, sample_count))
            P=P/P.sum()
            p_bias = p_bias_candidate[np.random.choice(sample_count, size=1, p=P).item()]
            
            grasp_pose=sapien.Pose()
            grasp_pose.set_p(handle_parent.get_handle_pcd_by_name(handle_name).mean(axis=0) + 
                             p_bias * handle_direction +
                             1*handle_radius * push_direction)

            sample_count = 1000
            circle_curve = lambda x : (1-(x-1)**2)**0.5
            P=-1e10*np.log(circle_curve(np.abs(np.linspace(-1,1, sample_count))))
            P=P/P.sum()
            q_x_bias_candidate = np.linspace(-np.pi/6, np.pi/6, sample_count)
            q_y_bias_candidate = np.linspace(-np.pi/6, np.pi/6, sample_count)
            q_z_bias_candidate = np.linspace(-np.pi/6, np.pi/6, sample_count)
            q_x_bias = q_x_bias_candidate[np.random.choice(sample_count, size=1, p=P).item()]
            q_y_bias = q_y_bias_candidate[np.random.choice(sample_count, size=1, p=P).item()]
            q_z_bias = q_z_bias_candidate[np.random.choice(sample_count, size=1, p=P).item()]
            
            base_z_direction = push_direction
            base_z_direction=base_z_direction/np.linalg.norm(base_z_direction)
            base_x_direction = np.cross([0,0,1], base_z_direction)
            base_x_direction=base_x_direction/np.linalg.norm(base_x_direction)
            base_y_direction = np.cross(base_z_direction, base_x_direction)
            base_y_direction=base_y_direction/np.linalg.norm(base_y_direction)

            R = euler.euler2mat(q_z_bias, q_y_bias, q_x_bias, axes='rzyx')

            grasp_x_direction = (R@base_x_direction.T).T
            grasp_y_direction = (R@base_y_direction.T).T
            grasp_z_direction = (R@base_z_direction.T).T
            grasp_pose.set_q(panda_xyz_direction_quant(grasp_x_direction, grasp_y_direction, grasp_z_direction))

            ungrasp_pose=sapien.Pose()
            ungrasp_pose.set_p(grasp_pose.p+push_direction*close_displacement)
            ungrasp_pose.set_q(grasp_pose.q)

            pregrasp_pose=sapien.Pose()
            pregrasp_pose.set_p(grasp_pose.p-
                                np.random.uniform(max(0.05, close_displacement/3), close_displacement)*push_direction + 
                                np.random.randn(3)/50)
            pregrasp_pose.set_q(euler.euler2quat(*(euler.quat2euler(grasp_pose.q)+np.random.randn(3)/50)))

            return pregrasp_pose, grasp_pose, ungrasp_pose

        assert "handle" in handle_name, "Only the handle of target object is graspable."

        self._open_gripper()

        pregrasp_pose, grasp_pose, ungrasp_pose = _compute_pose_for_grasp(handle_name, target_open_degree)
        if grasp_pose==None or pregrasp_pose==None or ungrasp_pose==None:
            return -1

        if self._move_to_pose(pregrasp_pose, collision_avoid_all=True)==-1:
            raise Exception()
            print("Collision Avoidance Computation Fails.")
            if self._move_to_pose(pregrasp_pose)==-1:
                print("Inverse Kinematics Computation Fails.")
                return -1
        
        if self._move_to_pose(grasp_pose)==-1:
            print("Inverse Kinematics Computation Fails.")
            return -1

        self._close_gripper()
        
        if self._move_to_pose(ungrasp_pose, speed_factor=0.3, guarantee_screw_mp=True)==-1:
            print("Inverse Kinematics Computation Fails.")
            return -1
        
        self._open_gripper()
        
        return 0
    

    def Press(self, ):
        pass