import sapien.core as sapien
from sapien.core import CameraEntity, Actor, Pose
from sapien.utils import Viewer
import numpy as np
from transforms3d import euler
from PIL import Image, ImageColor
import os
from typing import Tuple
import open3d as o3d

from utils import create_box, \
    create_capsule, \
    create_sphere, \
    create_table, \
    load_object_mesh, \
    load_robot
from action import PandaPrimitives
from config import manipulate_root_path, dataset_path
from perception import get_pcd_from_actor
from perception.scene_graph import SceneGraph, Node
from scene.core import TaskScene
from scene.specified_object import StorageFurniture
from utils import load_articulation


class SimplePickPlaceScene(TaskScene):
    def __init__(self):
        super().__init__()
        
        self._set_ground_texture(texture_file=os.path.join(manipulate_root_path, "assets/ground_texture_1.jpg"))

        self.subtask_dir = None
        self.step_index = 0

        self.backward_camera = self._set_camera(
            name="backward_camera",
            camera_pose_origin=np.array([1.8, 0, 0.5]),
            camera_pose_target=np.array([0,0,0]),
        )
        self.downward_camera = self._set_camera(
            name="downward_camera",
            camera_pose_origin=np.array([0.4,0,1.5]),
            camera_pose_target=np.array([0.4,0,0]),
        )
        self.rightward_camera = self._set_camera(
            name="rightward_camera",
            camera_pose_origin=np.array([0.6,-1.2,0.5]),
            camera_pose_target=np.array([0.6,0.3,0]),
        )
        self.leftward_camera = self._set_camera(
            name="leftward_camera",
            camera_pose_origin=np.array([0.6,1.2,0.5]),
            camera_pose_target=np.array([0.6,-0.3,0]),
        )

        self.viewer.toggle_axes(show=False)
        self.viewer.toggle_camera_lines(show=False)

        self.primitives = PandaPrimitives(
            self,
            robot=self.get_robot_by_name("panda_robot"),
        )


    def _set_ground_texture(self, texture_file:str):
        for actor in self.scene.get_all_actors():
            if actor.get_name()=="ground":
                self.scene.remove_actor(actor)
        render_material = self.renderer.create_material()
        render_texture = self.renderer.create_texture_from_file(filename=texture_file)
        render_material.set_diffuse_texture(render_texture)
        self.scene.add_ground(altitude=-0.45667, render_material=render_material)


    def get_object_list(self):
        return self.object_list
    

    def get_robot_list(self):
        return self.robot_list

    
    def _set_camera(
            self,
            name = "_camera",
            near:float = 0.05,
            far:float = 100,
            width:int = 640, 
            height:int = 480,
            fovy:np.ndarray = np.deg2rad(57.3),
            camera_pose_origin:np.ndarray = np.array([1.66, 0, 0.8]),
            camera_pose_target:np.ndarray = np.array([0, 0, 0]),
        )->CameraEntity:
        camera = self.scene.add_camera(
            name=name,
            width=width,
            height=height,
            fovy=fovy,
            near=near,
            far=far,
        )

        # Compute the camera pose by specifying forward(x), left(y) and up(z)
        forward = camera_pose_target - camera_pose_origin
        forward = forward / np.linalg.norm(forward)
        left = np.cross([0, 0, 1], forward)
        # In case the forward direction is parallel with the z axes.
        if np.linalg.norm(left)==0:
            # chose default left direction
            if np.array_equal(forward, np.array([0,0,-1])):
                left=np.array([0,1,0])
            elif np.array_equal(forward, np.array([0,0,1])):
                left=np.array([0,-1,0])
        else:
            left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = camera_pose_origin
        camera.set_pose(sapien.Pose.from_transformation_matrix(mat44))

        return camera
    
    def _mount_camera(
            self,
            camera_mount_actor: Actor,
            wrt_pose: Pose = Pose(),
            name = "_camera",
            near:float = 0.05,
            far:float = 100,
            width:int = 640, 
            height:int = 480,
            fovy:np.ndarray = np.deg2rad(57.3),
        )->CameraEntity:
        camera = self.scene.add_camera(
            name=name,
            width=width,
            height=height,
            fovy=fovy,
            near=near,
            far=far,
        )
        camera.set_pose(wrt_pose)
        camera.set_parent(parent=camera_mount_actor, keep_pose=False)

        return camera


    def _create_tabletop(self) -> None:
        # table top
        load_articulation(
            task_scene=self,
            urdf_file_path=os.path.join(manipulate_root_path, "assets/object/partnet-mobility/20985/mobility.urdf"),
            # pose=sapien.Pose(p=[0.46,0,-0.087578], q=euler.euler2quat(0,0,np.pi)),
            pose=sapien.Pose(p=[0.46,0,-0.097578], q=euler.euler2quat(0,0,np.pi)),
            name="table",
        )     

        # pads
        create_box(
            task_scene=self,
            pose=sapien.Pose([0.2, 0.58, 0.005]),
            half_size=[0.1, 0.1, 0.005],
            color=[1., 0., 0.],
            name='red_pad',
        )

        create_box(
            task_scene=self,
            pose=sapien.Pose([0.44, 0.58, 0.005]),
            half_size=[0.1, 0.1, 0.005],
            color=[0., 1., 0.],
            name='green_pad',
        )

        create_box(
            task_scene=self,
            pose=sapien.Pose([0.2, 0.35, 0.005]),
            half_size=[0.1, 0.1, 0.005],
            color=[0., 0., 1.],
            name='blue_pad',
        )

        #objects
        create_box(
            self,
            sapien.Pose(p=[0.46, -0.2, 0.06]),
            half_size=[0.05, 0.02, 0.06],
            color=[1., 0., 0.],
            name='box',
        )

        create_sphere(
            self,
            sapien.Pose(p=[0.66, 0.2, 0.02]),
            radius=0.02,
            color=[0., 1., 0.],
            name='sphere',
        )

        create_capsule(
            self,
            sapien.Pose(p=[0.3+0.2, 0.2, 0.02]),
            radius=0.02,
            half_length=0.03,
            color=[0., 0., 1.],
            name='capsule',
        )

        load_object_mesh(
            self, 
            self.renderer,
            sapien.Pose(p=[0.2+0.46, 0, 0.01865]), 
            collision_file_path=manipulate_root_path+'assets/object/banana/collision_meshes/collision.obj',
            visual_file_path=manipulate_root_path+'assets/object/banana/visual_meshes/visual.dae',
            name='banana',
        )


    def _create_robot(self) -> None:
        camera = self.scene.add_camera(
            name="first_person_camera",
            width=640,
            height=480,
            fovy=np.deg2rad(57.3),
            near=0.001,
            far=100,
        )
        mounted_camera_info={
            "camera_entity_link_name":"camera_entity",
            "camera_entity_joint_name":"camera_entity_joint",
            "camera_entity_half_size":[0.015, 0.045, 0.015],
            "camera_entity_pose":sapien.Pose(p=[0.04, 0, 0.055], q=euler.euler2quat(0,np.deg2rad(-70),np.pi)),
            "camera_entity_color":[201/255, 204/255, 207/255],
            "camera":camera,
            "camera_local_pose":sapien.Pose(p=[0.04, 0, 0.055], q=euler.euler2quat(0,np.deg2rad(-70),np.pi))
        }
        load_robot(
            task_scene=self,
            pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
            init_qpos=[0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0, 0],
            urdf_file_path=manipulate_root_path+"assets/robot/panda/panda.urdf",
            srdf_file_path=manipulate_root_path+"assets/robot/panda/panda.srdf",
            move_group="panda_hand",
            active_joints_num_wo_EE=7,
            name="panda_robot",
            mounted_camera_info=mounted_camera_info,
        )


    def get_scene_graph(self):
        self.scenegraph=SceneGraph()

        for obj in self.object_list:
            if type(obj)==sapien.Actor and obj.get_name()!="table":
                pcd = get_pcd_from_actor(obj)
                node=Node(obj.get_name(), pcd)
                self.scenegraph.add_node_wo_state(node)
            elif type(obj)==StorageFurniture:
                node=Node(obj.get_name(), obj.get_pcd())
                self.scenegraph.add_node_wo_state(node)
                for handle_name in obj.get_handle_name_list():
                    handle_pcd = obj.get_handle_pcd_by_name(handle_name)
                    node = Node(handle_name, handle_pcd)
                    self.scenegraph.add_node_wo_state(node)
                for drawer_name in obj.get_drawer_name_list():
                    drawer_pcd = obj.get_drawer_pcd_by_name(drawer_name)
                    handle = obj.get_handle_by_drawer(obj.get_drawer_by_name(drawer_name))
                    node = Node(drawer_name + \
                                f" with open degree: {obj.get_open_distance_by_name(handle.get_name())} / {obj.get_open_limit_by_name(handle.get_name()):.2f}", 
                                drawer_pcd)
                    self.scenegraph.add_node_wo_state(node)

        return self.scenegraph


    def demo(self, step = False) -> None:
        while not self.viewer.closed:
            if step:
                self.scene.step()
            self.scene.update_render()
            self.viewer.render()

    def set_subtask_dir(self, subtask_dir:str):
        self.subtask_dir=subtask_dir

    def set_step_index(self, step_index:int):
        self.step_index=step_index

    def step(
            self, 
            render_step: int = 1, 
            n_render_step: int = 1,
            backward_record :bool = True,
            downward_record :bool = False,
            rightward_record :bool = False,
            leftward_record :bool = False,
            first_person_record :bool = False,
    ):
        n_render_step = 1
        self.scene.step()

        def _get_RGB(camera: CameraEntity, viewpoint: str):
            assert viewpoint in ["Backward", "Downward", "Rightward", "Leftward", "FirstPerson"], \
                "The viewpoint should be \"Backward\", \"Downward\", \"Rightward\", \"Leftward\" or \"FirstPerson\""
            camera.take_picture()
            rgba = camera.get_float_texture("Color")
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            save_dir = os.path.join(self.subtask_dir, viewpoint, "RGB")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = str(self.step_index).zfill(5)+".png"
            rgba_pil.save(os.path.join(save_dir, save_file))
        def _get_Depth(camera: CameraEntity, viewpoint: str):
            assert viewpoint in ["Backward", "Downward", "Rightward", "Leftward", "FirstPerson"], \
                "The viewpoint should be \"Backward\", \"Downward\", \"Rightward\", \"Leftward\" or \"FirstPerson\""
            camera.take_picture()
            position = camera.get_float_texture('Position')
            depth = -position[..., 2]
            depth_img = (depth * 1000.0).astype(np.uint16)
            depth_pil = Image.fromarray(depth_img)
            save_dir = os.path.join(self.subtask_dir, viewpoint, "Depth")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = str(self.step_index).zfill(5)+".png"
            depth_pil.save(os.path.join(save_dir, save_file))

        if render_step % n_render_step == 0:
            self.scene.update_render()
            self.viewer.render()

            if backward_record:
                _get_RGB(self.backward_camera, "Backward")
                _get_Depth(self.backward_camera, "Backward")
            if downward_record:
                _get_RGB(self.downward_camera, "Downward")
                _get_Depth(self.downward_camera, "Downward")
            if rightward_record:
                _get_RGB(self.rightward_camera, "Rightward")
                _get_Depth(self.rightward_camera, "Rightward")
            if leftward_record:
                _get_RGB(self.leftward_camera, "Leftward")
                _get_Depth(self.leftward_camera, "Leftward")                
            if first_person_record:
                # for camera_entity in self.get_robot_by_name("panda_robot"):
                for camera in self.scene.get_cameras():
                    if camera.get_name()=='first_person_camera':
                        first_person_camera=camera
                _get_RGB(first_person_camera, "FirstPerson")
                _get_Depth(first_person_camera, "FirstPerson") 

        self.step_index+=1


if __name__ == '__main__':
    demo=SimplePickPlaceScene()
    # demo.demo()
    demo.scene.step() 
    demo.scene.update_render()

    demo.set_subtask_dir(os.path.join(dataset_path, "task_0001/subtask_001"))
    demo.set_step_index(0)
    demo.primitives.Push("box", [0,-1], 0.1)
    demo.set_step_index(0)