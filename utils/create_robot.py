import sapien.core as sapien
import mplib
import numpy as np
from sapien.utils.viewer import Viewer
from transforms3d import euler
from scene.core import TaskScene
from scene.specified_object import Robot
from typing import TypedDict

from perception.core import get_pcd_from_actor


class MountCameraInfo(TypedDict):
    camera_entity_link_name:str
    camera_entity_joint_name:str
    camera_entity_half_size:list
    camera_entity_pose:sapien.Pose
    camera_entity_color:list
    camera:sapien.CameraEntity
    camera_local_pose:sapien.Pose


def load_robot(
        task_scene: TaskScene,
        pose: sapien.Pose,
        init_qpos: list,
        urdf_file_path: str,
        srdf_file_path:str,
        move_group:str,
        active_joints_num_wo_EE:int,
        fix_root_link=True,
        uniform_stiffness=1000,
        uniform_damping=200,
        name='',
        mounted_camera_info:MountCameraInfo={},
) -> Robot:
    # Robot
    # Load URDF
    loader: sapien.URDFLoader = task_scene.scene.create_urdf_loader()
    robot_builder: sapien.ArticulationBuilder = loader.load_file_as_articulation_builder(urdf_file_path)

    # Set the mounted camera entity
    if mounted_camera_info:
        for link_builder in robot_builder.get_link_builders():
            if link_builder.get_name()==move_group:
                robot_move_group_builder = link_builder
                break
        camera_entity_builder = robot_builder.create_link_builder(robot_move_group_builder)
        camera_entity_builder.set_name(mounted_camera_info["camera_entity_link_name"])
        camera_entity_builder.set_joint_name(mounted_camera_info["camera_entity_joint_name"])
        camera_entity_builder.add_box_collision(pose=mounted_camera_info["camera_entity_pose"], 
                                                half_size=mounted_camera_info["camera_entity_half_size"],
                                                density=1000)
        camera_entity_builder.add_box_visual(half_size=mounted_camera_info["camera_entity_half_size"], 
                                            pose=mounted_camera_info["camera_entity_pose"], 
                                            color=mounted_camera_info["camera_entity_color"])
        camera_entity_builder.set_joint_properties(joint_type="fixed",limits=[])

    # Build the robot articulation
    robot=robot_builder.build(fix_root_link=fix_root_link)
    robot.set_root_pose(pose)
    robot.set_name(name=name)

    # Set initial joint positions
    robot.set_qpos(init_qpos)

    # Used for PID control
    active_joints = robot.get_active_joints()
    for joint in active_joints:
        joint.set_drive_property(stiffness=uniform_stiffness, damping=uniform_damping)

    # Mount the camera
    if mounted_camera_info:
        for link in robot.get_links():
            if link.get_name()==mounted_camera_info["camera_entity_link_name"]:
                mounted_camera_info["camera"].set_parent(parent=link, keep_pose=False)      
                mounted_camera_info["camera"].set_local_pose(mounted_camera_info["camera_local_pose"])   
                break

    robot_specified_object=Robot(
        task_scene,
        robot_articulation=robot,
        urdf_file_path=urdf_file_path,
        srdf_file_path=srdf_file_path,
        move_group=move_group,
        active_joints_num_wo_EE=active_joints_num_wo_EE,
    )
    if mounted_camera_info:
        robot_specified_object.add_mounted_obj(mounted_camera_info["camera"].parent)
    task_scene.robot_list.append(robot_specified_object)
    task_scene.object_list

    return robot_specified_object