"""Create actors (rigid bodies).

The actor (or rigid body) in Sapien is created through a sapien.ActorBuilder.
An actor consists of both collision shapes (for physical simulation) and visual shapes (for rendering).
Note that an actor can have multiple collision and visual shapes,
and they do not need to correspond.

Concepts:
    - Create sapien.Actor by primitives (box, sphere, capsule)
    - Create sapien.Actor by mesh files
    - sapien.Pose
"""

import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
from transforms3d import euler

from config import *
from scene.core import TaskScene, SpecifiedObject


def create_box(
        task_scene: TaskScene,
        pose: sapien.Pose,
        half_size=[0.05, 0.05, 0.05],
        color=[1., 0., 0.],
        name='',
        load_in = True,
) -> sapien.Actor:
    """Create a box.

    Args:
        scene: sapien.Scene to create a box.
        pose: 6D pose of the box.
        half_size: [3], half size along x, y, z axes.
        color: [3] or [4], rgb or rgba
        name: name of the actor.

    Returns:
        sapien.Actor
    """
    scene = task_scene.scene
    half_size = np.array(half_size)
    builder: sapien.ActorBuilder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size)  # Add collision shape
    builder.add_box_visual(half_size=half_size, color=color)  # Add visual shape
    box: sapien.Actor = builder.build(name=name)

    # Or you can set_name after building the actor
    # box.set_name(name)
    box.set_pose(pose)

    if load_in:
        task_scene.object_list.append(box)
    
    return box


def create_sphere(
        task_scene: TaskScene,
        pose: sapien.Pose,
        radius=0.05,
        color=[0., 1., 0.],
        name='',
        load_in = True,
) -> sapien.Actor:
    """Create a sphere. See create_box."""
    builder = task_scene.scene.create_actor_builder()
    builder.add_sphere_collision(radius=radius)
    builder.add_sphere_visual(radius=radius, color=color)
    sphere = builder.build(name=name)
    sphere.set_pose(pose)
    # NOTE: Since Sapien does not model rolling resistance (friction), the sphere will roll forever.
    # However, you can set actor's damping, like air resistance.
    sphere.set_damping(linear=20, angular=20)

    if load_in:
        task_scene.object_list.append(sphere)

    return sphere


def create_capsule(
        task_scene: TaskScene,
        pose: sapien.Pose,
        radius=0.05,
        half_length=0.05,
        color=[0., 0., 1.],
        name='',
        load_in = True,
) -> sapien.Actor:
    """Create a capsule (x-axis <-> half_length). See create_box."""
    builder = task_scene.scene.create_actor_builder()
    builder.add_capsule_collision(radius=radius, half_length=half_length)
    builder.add_capsule_visual(radius=radius, half_length=half_length, color=color)
    capsule = builder.build(name=name)
    capsule.set_pose(pose)
    # NOTE: Since Sapien does not model rolling resistance (friction), the sphere will roll forever.
    # However, you can set actor's damping, like air resistance.
    capsule.set_damping(linear=20, angular=20)

    if load_in:
        task_scene.object_list.append(capsule)

    return capsule


def create_table(
        task_scene: TaskScene,
        pose: sapien.Pose,
        size=[1., 1.],
        height=1.0,
        thickness=0.1,
        color=(0.8, 0.6, 0.4),
        name='table',
        load_in = True,
) -> sapien.Actor:
    """Create a table (a collection of collision and visual shapes)."""
    builder = task_scene.scene.create_actor_builder()
    
    # Tabletop
    tabletop_half_size = [size[0] / 2, size[1] / 2, thickness / 2]
    builder.add_box_collision(half_size=tabletop_half_size)
    builder.add_box_visual(half_size=tabletop_half_size, color=color)
    
    # Table legs (x4)
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (size[0] - thickness) / 2
            y = j * (size[1] - thickness) / 2
            table_leg_pose = sapien.Pose([x, y, -height / 2])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2]
            builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
            builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, color=color)

    table = builder.build_kinematic(name=name)
    pose.set_p(np.array([pose.p[0], pose.p[1], pose.p[2]-thickness / 2])) # Make the top surface's z equal to 0
    table.set_pose(pose)

    if load_in:
        task_scene.object_list.append(table)

    return table


def load_object_mesh(
        task_scene: TaskScene,
        pose: sapien.Pose,
        collision_file_path='',
        visual_file_path='',
        texture_file_path='',
        scale = np.array([1., 1., 1.]),
        name='',
        is_kinematic=False,
        load_in = True,
) -> sapien.Actor:
    if texture_file_path:
        material = task_scene.renderer.create_material()
        material.base_color = [1.0, 1.0, 1.0, 1.0]
        material.diffuse_texture_filename = texture_file_path
        material.metallic = 0.001
        material.roughness = 0.4
    else:
        material = None

    builder = task_scene.scene.create_actor_builder()
    # Any collision shape in SAPIEN is required to be convex. 
    # To this end, a mesh will be “cooked” into a convex mesh before being used in the simulation.
    # HOWEVER， you can still add a nonconvex collision shape from a file. 
    # If it is not a trigger, then it is only valid for static and kinematic actors.
    if is_kinematic:
        builder.add_nonconvex_collision_from_file(filename=collision_file_path, scale=scale)
        builder.add_visual_from_file(filename=visual_file_path, scale=scale, material=material)

        mesh = builder.build_kinematic(name=name)
        mesh.set_pose(pose)
    else:
        builder.add_multiple_collisions_from_file(filename=collision_file_path, scale=scale)
        builder.add_visual_from_file(filename=visual_file_path, scale=scale, material=material)

        mesh = builder.build(name=name)
        mesh.set_pose(pose)

    if load_in:
        task_scene.object_list.append(mesh)
    
    return mesh


def load_articulation(
        task_scene: TaskScene,
        urdf_file_path: str,
        scale=1,
        pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
        name='',
        load_in = True,
) -> sapien.Articulation:
    loader = task_scene.scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link=False

    model = loader.load(urdf_file_path)
    model.set_pose(pose=pose)
    model.set_name(name=name)

    # for joint in model.get_active_joints():
    #     joint.set_drive_property(stiffness=0, damping=20)

    if load_in:
        task_scene.object_list.append(model)

    return model


def load_specified_object(
        task_scene: TaskScene,
        specified_object: SpecifiedObject,
        load_in = True,
) -> SpecifiedObject:

    if load_in:
        specified_object.load_in(task_scene)
        # task_scene.object_list.append(specified_object)

    return specified_object


def main():
    task_scene = TaskScene()
    scene = task_scene.scene
    renderer = task_scene.renderer

    # ---------------------------------------------------------------------------- #
    # Add actors
    # ---------------------------------------------------------------------------- #
    task_scene.scene.add_ground(altitude=0)  # The ground is in fact a special actor.
    box = create_box(
        task_scene,
        sapien.Pose(p=[0, 0, 1.0 + 0.05]),
        half_size=[0.05, 0.05, 0.05],
        color=[1., 0., 0.],
        name='box',
    )
    sphere = create_sphere(
        task_scene,
        sapien.Pose(p=[0, -0.2, 1.0 + 0.05]),
        radius=0.05,
        color=[0., 1., 0.],
        name='sphere',
    )
    capsule = create_capsule(
        task_scene,
        sapien.Pose(p=[0, 0.2, 1.0 + 0.05]),
        radius=0.05,
        half_length=0.05,
        color=[0., 0., 1.],
        name='capsule',
    )
    table = create_table(
        task_scene,
        sapien.Pose(p=[0, 0, 1.0]),
        size=[1.0, 1.0],
        height=1.0,
    )

    # add a mesh
    mesh = load_object_mesh(
        task_scene, 
        sapien.Pose(p=[-0.2, 0, 1.0 + 0.05]), 
        collision_file_path=manipulate_root_path+'assets/object/banana/collision_meshes/collision.obj',
        visual_file_path=manipulate_root_path+'assets/object/banana/visual_meshes/visual.dae',
        name='mesh',
        )
    

    # ---------------------------------------------------------------------------- #
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = task_scene.viewer

    viewer.set_camera_xyz(x=-2, y=0, z=2.5)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    task_scene.demo()


if __name__ == '__main__':
    main()