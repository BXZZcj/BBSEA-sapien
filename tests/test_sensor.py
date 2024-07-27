import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
from transforms3d import euler

from utils import create_box, \
    create_capsule, \
    create_sphere, \
    create_table, \
    load_object_mesh, \
    load_robot
from action import PandaPrimitives
from config import manipulate_root_path
from perception import get_pcd_from_actor
from perception.scene_graph import SceneGraph, Node
from scene.core import TaskScene
from scene.specified_object import StorageFurniture
from utils import load_partnet_mobility


class SimplePickPlaceScene(TaskScene):
    def __init__(self):
        super().__init__()

        self.primitives = PandaPrimitives(
            self,
            robot=self.panda_robot,
            time_step=self.timestep,
        )


    def get_object_list(self):
        return self.object_list
    

    def get_robot_list(self):
        return self.robot_list


    def _create_tabletop(self) -> None:
        # table top
        self.table=create_table(
            task_scene=self,
            pose=sapien.Pose([0.57, 0, 0]),
            size=[1., 1.4],
            height=1,
            thickness=0.1,
            name="table",
            )

        # pads
        self.red_pad=create_box(
            task_scene=self,
            pose=sapien.Pose([0.2, 0.58, 0.005]),
            half_size=[0.1, 0.1, 0.005],
            color=[1., 0., 0.],
            name='red_pad',
        )

        self.green_pad=create_box(
            task_scene=self,
            pose=sapien.Pose([0.44, 0.58, 0.005]),
            half_size=[0.1, 0.1, 0.005],
            color=[0., 1., 0.],
            name='green_pad',
        )

        self.blue_pad=create_box(
            task_scene=self,
            pose=sapien.Pose([0.2, 0.35, 0.005]),
            half_size=[0.1, 0.1, 0.005],
            color=[0., 0., 1.],
            name='blue_pad',
        )

        #objects
        self.box = create_box(
            self,
            sapien.Pose(p=[0.46, -0.1, 0.06]),
            half_size=[0.05, 0.02, 0.06],
            color=[1., 0., 0.],
            name='box',
        )

        self.sphere = create_sphere(
            self,
            sapien.Pose(p=[0.66, 0.2, 0.02]),
            radius=0.02,
            color=[0., 1., 0.],
            name='sphere',
        )

        self.capsule = create_capsule(
            self,
            sapien.Pose(p=[0.3+0.2, 0.2, 0.02]),
            radius=0.02,
            half_length=0.03,
            color=[0., 0., 1.],
            name='capsule',
        )

        self.banana = load_object_mesh(
            self, 
            self.renderer,
            sapien.Pose(p=[0.2+0.46, 0, 0.01865]), 
            collision_file_path=manipulate_root_path+'assets/object/banana/collision_meshes/collision.obj',
            visual_file_path=manipulate_root_path+'assets/object/banana/visual_meshes/visual.dae',
            name='banana',
        )


    def _create_robot(self) -> None:
        # Robot
        # Load URDF
        self.panda_robot=load_robot(
            task_scene=self,
            pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
            init_qpos=[0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0, 0],
            urdf_file_path=manipulate_root_path+"assets/robot/panda/panda.urdf",
            srdf_file_path=manipulate_root_path+"assets/robot/panda/panda.srdf",
            move_group="panda_hand",
            active_joints_num_wo_MG=7,
            name="panda_robot",
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

    def step(
            self, 
            render_step: int = 1, 
            n_render_step: int = 1
    ):
        pass



if __name__ == '__main__':
    demo=SimplePickPlaceScene()
    demo.scene.step() 
    demo.scene.update_render()

    demo.primitives.Push("box",[0,-1],0.1)
    # demo.primitives.Pick("banana")
    # demo.primitives.PlaceOn("green_pad")