import sapien.core as sapien
from sapien.core import CameraEntity, Actor, Pose
from sapien.utils import Viewer
import numpy as np
from transforms3d import euler
from PIL import Image, ImageColor
import os
from typing import Tuple
import open3d as o3d

from utils import load_object_mesh, load_robot, load_specified_object
from action import PandaPrimitives
from config import manipulate_root_path, dataset_path
from perception import get_pcd_from_actor, get_pcd_from_obj
from perception.scene_graph import SceneGraph, Node
from scene.core import TaskScene
from scene.specified_object import StorageFurniture, Catapult
from utils import load_articulation


class SimplePickPlaceScene(TaskScene):
    def __init__(self):
        super().__init__()
        
        self._set_ground(
            altitude=-0.44867,
            # altitude=-0.46867,
            texture_file=os.path.join(manipulate_root_path, "assets/ground_texture_1.jpg"),
        )

        self.task_index=None
        self.subtask_index=None
        self.subtask_dir = None
        self.step_index = 0
        self.step_callback=lambda:None

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

        def step_callback():
            if self.get_object_by_name("catapult").check_activate():
                self.get_object_by_name("catapult").activate_behavior()
        self.set_step_callback(step_callback)

        # self.viewer.set_camera_xyz(x=1.56, y=-0.6, z=0.7)
        # self.viewer.set_camera_rpy(r=0, p=-0.8, y=np.pi)
        # self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)


    def _create_tabletop(self) -> None:
        # table top
        load_articulation(
            task_scene=self,
            urdf_file_path=os.path.join(manipulate_root_path, "assets/object/partnet-mobility/20985/mobility.urdf"),
            pose=sapien.Pose(p=[0.46,0,-0.087578], q=euler.euler2quat(0,0,np.pi)),
            name="table_20985",
        ) 
        # self.get_object_by_name("StorageFurniture45290").set_open_degree_by_name("StorageFurniture45290_handle_2", 0.5)
        load_specified_object(
            self,
            Catapult(
                load_articulation(
                    task_scene=self,
                    urdf_file_path=os.path.join(manipulate_root_path, "assets/object/catapult/catapult.urdf"),
                    scale=1,
                    pose=sapien.Pose(p=[0.5,0.1,0.0130003]),
                    name="catapult",
                    load_in=False,
                ),
                initial_qf=np.array([-1,0])
            ),
            load_in=True,
        )

        load_object_mesh(
            self, 
            sapien.Pose(p=[0.66, 0.05, 0.0187573]), 
            collision_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/058_golf_ball/textured.obj'),
            visual_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/058_golf_ball/textured.obj'),
            texture_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/058_golf_ball/texture_map.png'),
            name='058_golf_ball',
        )
        load_object_mesh(
            self, 
            sapien.Pose(p=[0.66, 0, 0.0163004], q=euler.euler2quat(0,0,0)), 
            collision_file_path=os.path.join(manipulate_root_path, 'assets/object/banana/collision_meshes/collision.obj'),
            visual_file_path=os.path.join(manipulate_root_path, 'assets/object/banana/visual_meshes/visual.dae'),
            name='banana',
        )
        load_object_mesh(
            self, 
            sapien.Pose(p=[0.75, 0.1, 0.0266351]), 
            collision_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/077_rubiks_cube/textured.obj'),
            visual_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/077_rubiks_cube/textured.obj'),
            texture_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/077_rubiks_cube/texture_map.png'),
            name='077_rubiks_cube',
        )
        # load_object_mesh(
        #     self, 
        #     sapien.Pose(p=[0.541724, -0.287774, 0.027222]), 
        #     collision_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-g_lego_duplo/textured.obj'),
        #     visual_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-g_lego_duplo/textured.obj'),
        #     texture_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-g_lego_duplo/texture_map.png'),
        #     name='073-g_lego_duplo',
        # )
        load_object_mesh(
            self, 
            sapien.Pose(p=[0.406019, -0.340739, 0.0195762]), 
            collision_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-f_lego_duplo/textured.obj'),
            visual_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-f_lego_duplo/textured.obj'),
            texture_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-f_lego_duplo/texture_map.png'),
            name='073-f_lego_duplo',
        )
        load_object_mesh(
            self, 
            sapien.Pose(p=[0.505376, -0.311012, 0.0727183]), 
            collision_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-e_lego_duplo/textured.obj'),
            visual_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-e_lego_duplo/textured.obj'),
            texture_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-e_lego_duplo/texture_map.png'),
            name='073-e_lego_duplo',
        )
        load_object_mesh(
            self, 
            sapien.Pose(p=[0.369825, -0.301168, 0.0194148]), 
            collision_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-d_lego_duplo/textured.obj'),
            visual_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-d_lego_duplo/textured.obj'),
            texture_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-d_lego_duplo/texture_map.png'),
            name='073-d_lego_duplo',
        )
        load_object_mesh(
            self, 
            sapien.Pose(p=[0.490428, -0.499637, 0.00965894]), 
            collision_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-c_lego_duplo/textured.obj'),
            visual_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-c_lego_duplo/textured.obj'),
            texture_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-c_lego_duplo/texture_map.png'),
            name='073-c_lego_duplo',
        )
        load_object_mesh(
            self, 
            sapien.Pose(p=[0.630726, -0.412977, 0.0193675]), 
            collision_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-b_lego_duplo/textured.obj'),
            visual_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-b_lego_duplo/textured.obj'),
            texture_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-b_lego_duplo/texture_map.png'),
            name='073-b_lego_duplo',
        )
        load_object_mesh(
            self, 
            sapien.Pose(p=[0.355787, -0.249862, 0.00983431]), 
            collision_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-a_lego_duplo/textured.obj'),
            visual_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-a_lego_duplo/textured.obj'),
            texture_file_path=os.path.join(manipulate_root_path, 'assets/object/mani_skill2_ycb/models/073-a_lego_duplo/texture_map.png'),
            name='073-a_lego_duplo',
        )
        

    def _create_robot(self) -> None:
        self.first_person_camera = self._set_camera(
            name="first_person_camera",
        )
        mounted_camera_info={
            "camera_entity_link_name":"camera_entity",
            "camera_entity_joint_name":"camera_entity_joint",
            "camera_entity_half_size":[0.015, 0.045, 0.015],
            "camera_entity_pose":sapien.Pose(p=[0.04, 0, 0.055], q=euler.euler2quat(0,np.deg2rad(-70),np.pi)),
            "camera_entity_color":[201/255, 204/255, 207/255],
            "camera":self.first_person_camera,
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
            if type(obj)==sapien.Actor or type(obj)==sapien.Articulation:# and obj.get_name()!="table":
                pcd = get_pcd_from_obj(obj)
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
    
    def set_task_index(self, task_index:int, subtask_index:int=None):
        self.task_index=task_index
        if subtask_index!=None:
            self.subtask_index=subtask_index
    
    def get_task_dir(self):
        dir = os.path.join(dataset_path, f"task_{self.task_index:04}")
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    
    def get_subtask_dir(self):
        dir = os.path.join(dataset_path, f"task_{self.task_index:04}",f"subtask_{self.subtask_index:03}")
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    
    def set_step_index(self, step_index:int):
        self.step_index=step_index

    def set_step_callback(self, step_callback):
        self.step_callback=step_callback

    def step(
            self, 
            render_step: int = 1, 
            n_render_step: int = 1,
            backward_record :bool = False,
            downward_record :bool = False,
            rightward_record :bool = False,
            leftward_record :bool = False,
            first_person_record :bool = False,
    ):
        n_render_step = 8
        self.scene.step()
        self.step_callback()

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
                _get_RGB(self.first_person_camera, "FirstPerson")
                _get_Depth(self.first_person_camera, "FirstPerson") 

        self.step_index+=1
    

    def demo(self, step = False) -> None:
        while not self.viewer.closed:
            if step:
                self.scene.step() 
            # print(self.catapult.get_qlimits())
            # print(self.get_object_by_name("catapult").get_pose())
            self.scene.update_render()
            self.viewer.render()


if __name__ == '__main__':
    demo=SimplePickPlaceScene()
    # demo.demo(step=False)
    # print(demo.get_scene_graph())
    
    demo.set_task_index(1, 1)

    demo.set_step_index(0)
    demo.primitives.Pick('banana')
    demo.primitives.PlaceOn('073-a_lego_duplo')
    # try:
    #     demo.primitives.Pick('073-g_lego_duplo')
    #     demo.primitives.PlaceOn('073-a_lego_duplo')
    # except Exception as e:
    #     print(f"A error occurred: {e}")
    demo.set_step_index(0)
    
    # demo.demo(step=True)