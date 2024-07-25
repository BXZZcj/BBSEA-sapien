import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np

from perception import get_actor_names_in_scene, \
    get_actor_by_name, \
    get_pcd_from_actor
from perception.scene_graph import SceneGraph, Node


class TaskScene():
    def __init__(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        self.scene = self.engine.create_scene(sapien.SceneConfig())
        self.timestep = 1 / 100.0
        self.scene.set_timestep(1 / 100.0)
        self.scene.add_ground(-1)
        self.scene.default_physical_material = self.scene.create_physical_material(static_friction=1, dynamic_friction=1, restitution=0.0)
        self.scene.set_ambient_light(color=[0.5, 0.5, 0.5])
        self.scene.add_directional_light(direction=[0, 1, -1], color=[0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light(position=[1, 2, 2], color=[1, 1, 1], shadow=True)
        self.scene.add_point_light(position=[1, -2, 2], color=[1, 1, 1], shadow=True)
        self.scene.add_point_light(position=[-1, 0, 1], color=[1, 1, 1], shadow=True)

        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(x=1.66, y=0, z=0.8)
        self.viewer.set_camera_rpy(r=0, p=-0.8, y=np.pi)
        self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self.object_list=[]
        self.robot_list=[]
        self._create_tabletop()
        self._create_robot()

        self.primitives = None


    def get_object_list(self):
        return self.object_list
    

    def get_robot_list(self):
        return self.robot_list


    def _create_tabletop(self) -> None:
        pass

    def _create_robot(self) -> None:
        pass

    def get_scene_graph(self):
        self.scenegraph=SceneGraph()
        actor_names = get_actor_names_in_scene(scene=self.scene)
        # eliminate the "ground" actor
        actor_names = [actor_name for actor_name in actor_names if actor_name != "ground"]

        for name in actor_names:
            actor = get_actor_by_name(scene=self.scene, name=name)
            pcd = get_pcd_from_actor(actor)
            if actor.get_builder().get_visuals()[0].type=="Box":
                pcd = dense_sample_convex_pcd(pcd)
            node=Node(name, pcd)
            self.scenegraph.add_node_wo_state(node)

        return self.scenegraph


    def demo(self, step = False) -> None:
        while not self.viewer.closed:
            if step:
                self.scene.step()
            self.scene.update_render()
            self.viewer.render()

if __name__ == '__main__':
    demo = TaskScene()
    demo.demo(step=False)
    print(demo.get_scene_graph())