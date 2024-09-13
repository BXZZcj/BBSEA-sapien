import sapien.core as sapien
from sapien.utils import Viewer
from sapien.core import Pose, Actor, Articulation, CameraEntity, Link
import numpy as np
from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    from scene.specified_object import Robot
    from scene.core import TaskScene, SpecifiedObject


class SpecifiedObject:
    def __init__(self):
        self.body: Union[Articulation, Link, Actor]=None
        self.parent: 'SpecifiedObject' = None
        pass

    def get_name(self)->str:
        return self.body.get_name()

    def get_pose(self)->sapien.Pose:
        return self.body.get_pose()
    
    def get_parent(self)->'SpecifiedObject':
        return self.parent

    def get_pcd(self)->np.ndarray:   
        from perception.core import get_pcd_from_obj
        return get_pcd_from_obj(self.body)

    def get_pcd_normals(self)->np.ndarray:
        from perception.core import get_pcd_normals_from_obj
        return get_pcd_normals_from_obj(self.body)

    def load_in(self, task_scene:'TaskScene'):
        task_scene.object_list.append(self)


class TaskScene():
    def __init__(
            self, 
            ray_tracking: bool = False,
            rt_samples_per_pixel: int=8,
            rt_use_denoiser:bool=True,
    ):
        self.engine = sapien.Engine()

        if ray_tracking:
            sapien.render_config.camera_shader_dir = "rt"
            sapien.render_config.viewer_shader_dir = "rt"
            sapien.render_config.rt_samples_per_pixel = rt_samples_per_pixel
            sapien.render_config.rt_use_denoiser = rt_use_denoiser

        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        self.scene = self.engine.create_scene(sapien.SceneConfig())
        self.time_step = 1 / 300.0
        self.scene.set_timestep(self.time_step)
        self._set_ground(altitude=-1)
        self.scene.default_physical_material = self.scene.create_physical_material(static_friction=1, dynamic_friction=1, restitution=0.0)
        self.scene.set_ambient_light(color=[0.5, 0.5, 0.5])
        self.scene.add_directional_light(direction=[0, 1, -1], color=[0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light(position=[1, 2, 2], color=[1, 1, 1], shadow=True)
        self.scene.add_point_light(position=[1, -2, 2], color=[1, 1, 1], shadow=True)
        self.scene.add_point_light(position=[-1, 0, 1], color=[1, 1, 1], shadow=True)

        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        # self.viewer.set_camera_xyz(x=0.26, y=-0.3, z=1.2)
        # self.viewer.set_camera_rpy(r=0, p=-np.pi/2, y=np.pi)
        self.viewer.set_camera_xyz(x=1.66, y=0, z=0.8)
        self.viewer.set_camera_rpy(r=0, p=-0.8, y=np.pi)
        self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self.object_list:list[Actor, Link, SpecifiedObject]=[]
        self.robot_list:list[Robot]=[]
        self._create_tabletop()
        self._create_robot()

        self.primitives = None
    
        # Formally initialize the environment
        self.scene.step() 
        self.scene.update_render()


    def get_robot_list(self):
        return self.robot_list

    def get_object_list(self)->list:
        return self.object_list
    

    def _set_ground(
            self, 
            altitude:int=-1,
            texture_file:str=None
    ):
        self.ground_altitude=altitude

        for actor in self.scene.get_all_actors():
            if actor.get_name()=="ground":
                self.scene.remove_actor(actor)

        if not texture_file:
            self.scene.add_ground(altitude=self.ground_altitude)
        else:
            render_material = self.renderer.create_material()
            render_texture = self.renderer.create_texture_from_file(filename=texture_file)
            render_material.set_diffuse_texture(render_texture)
            self.scene.add_ground(altitude=self.ground_altitude, render_material=render_material)
    
    
    def _set_camera(
            self,
            name = "_camera",
            near:float = 0.001,
            far:float = 100,
            width:int = 1024, 
            height:int = 768,
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
            width:int = 1024, 
            height:int = 768,
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


    def get_object_by_name(self, name)->Union[sapien.Actor,SpecifiedObject]:
        for obj in self.object_list:
            if obj.get_name()==name:
                return obj
        return None
    

    def get_robot_list(self)->list:
        return self.robot_list
    

    def get_robot_by_name(self, name) -> 'Robot':
        for robot in self.robot_list:
            if robot.get_name() == name:
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