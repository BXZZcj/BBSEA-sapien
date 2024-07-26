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
            viewer=self.viewer,
            robot=self.panda_robot, 
            urdf_file_path=self.panda_robot_urdf_path,
            srdf_file_path=self.panda_robot_srdf_path,
            gripper=self.move_group,
            time_step=self.timestep,
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7),
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

        # self.beer_can = load_object_mesh(
        #     self, 
        #     self.renderer,
        #     sapien.Pose(p=[0.2, 0.58, 0.0874323]), 
        #     collision_file_path=manipulate_root_path+"assets/object/beer_can/visual_mesh.obj",
        #     visual_file_path=manipulate_root_path+"assets/object/beer_can/visual_mesh.obj",
        #     texture_file_path=manipulate_root_path+"assets/object/beer_can/texture.png",
        #     name='beer_can',
        # )

        # self.pepsi_bottle = load_object_mesh(
        #     self, 
        #     self.renderer,
        #     sapien.Pose(p=[0.44, 0.58, 0.129699]), 
        #     collision_file_path=manipulate_root_path+"assets/object/pepsi_bottle/visual_mesh.obj",
        #     visual_file_path=manipulate_root_path+"assets/object/pepsi_bottle/visual_mesh.obj",
        #     texture_file_path=manipulate_root_path+"assets/object/pepsi_bottle/texture.png",
        #     name='pepsi_bottle',
        # )

        # self.champagne = load_object_mesh(
        #     self,
        #     self.renderer,
        #     sapien.Pose(p=[0.2, 0.35, 0.104486]), 
        #     collision_file_path=manipulate_root_path+"assets/object/champagne/visual_mesh.obj",
        #     visual_file_path=manipulate_root_path+"assets/object/champagne/visual_mesh.obj",
        #     texture_file_path=manipulate_root_path+"assets/object/champagne/texture.png",
        #     name='champagne',
        # )

        # Load the StorageFurniture into the Task Scene
        self.StorageFurniture45290 = StorageFurniture(
            load_partnet_mobility(
                task_scene=self,
                urdf_file_path=manipulate_root_path+"assets/object/partnet-mobility/45290/mobility.urdf",
                scale=0.3,
                pose=sapien.Pose([0.56, -0.45, 0.240578], euler.euler2quat(0,0,-np.pi/2)),
                name="StorageFurniture45290"
            )
        )
        self.object_list.append(self.StorageFurniture45290)


    def _create_robot(self) -> None:
        self.panda_robot_urdf_path=manipulate_root_path+"assets/robot/panda/panda.urdf"
        self.panda_robot_srdf_path=manipulate_root_path+"assets/robot/panda/panda.srdf"
        self.move_group="panda_hand"
        # Robot
        # Load URDF
        self.init_qpos=[0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
        self.panda_robot=load_robot(
            task_scene=self,
            pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
            init_qpos=self.init_qpos,
            urdf_file_path=self.panda_robot_urdf_path,
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


if __name__ == '__main__':
    demo = SimplePickPlaceScene()
    demo.demo(step=False)
    print(demo.get_scene_graph())