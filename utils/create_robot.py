import sapien.core as sapien
import mplib
import numpy as np
from sapien.utils.viewer import Viewer

from scene.core import TaskScene
from scene.specified_object import Robot


def load_robot(
        task_scene: TaskScene,
        pose: sapien.Pose,
        init_qpos: list,
        urdf_file_path: str,
        srdf_file_path:str,
        move_group:str,
        active_joints_num_wo_MG:int,
        fix_root_link=True,
        uniform_stiffness=1000,
        uniform_damping=200,
        name='',
) -> Robot:
    # Robot
    # Load URDF
    scene = task_scene.scene

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    robot: sapien.Articulation = loader.load(urdf_file_path)
    robot.set_root_pose(pose)
    robot.set_name(name=name)

    # Set initial joint positions
    robot.set_qpos(init_qpos)

    # Used for PID control
    active_joints = robot.get_active_joints()
    for joint in active_joints:
        joint.set_drive_property(stiffness=uniform_stiffness, damping=uniform_damping)

    robot_specified_object=Robot(
        robot_articulation=robot,
        urdf_file_path=urdf_file_path,
        srdf_file_path=srdf_file_path,
        move_group=move_group,
        active_joints_num_wo_MG=active_joints_num_wo_MG,
    )
    task_scene.robot_list.append(robot_specified_object)

    return robot_specified_object