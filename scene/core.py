import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
from typing import Union


class SpecifiedObject:
    def __init__(self):
        pass

    def get_name(self):
        pass

    def get_pose(self)->sapien.Pose:
        pass

    def get_pcd(self)->np.ndarray:
        pass

    def get_pcd_normals(self)->np.ndarray:
        pass


class TaskScene():
    def __init__(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        self.scene = self.engine.create_scene(sapien.SceneConfig())
        self.time_step = 1 / 100.0
        self.scene.set_timestep(self.time_step)
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


    def get_object_list(self)->list:
        return self.object_list


    def get_object_by_name(self, name)->Union[sapien.Actor,SpecifiedObject]:
        for obj in self.object_list:
            if obj.get_name()==name:
                return obj
        return None
    

    def get_robot_list(self)->list:
        return self.robot_list
    

    def get_robot_by_name(self, name)->sapien.Articulation:
        for robot in self.robot_list:
            if robot.get_name()==name:
                return robot
        return None


    def _create_tabletop(self) -> None:
        pass

    def _create_robot(self) -> None:
        pass

    def step(
            self, 
            render_step:int = 1, 
            n_render_step:int = 1,
        ) -> None:
        self.scene.step()
        if render_step % n_render_step == 0:
            self.scene.update_render()
            self.viewer.render()


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