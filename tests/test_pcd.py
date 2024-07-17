import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
from transforms3d.euler import euler2quat

from utils import *
from api import *
from config import *


#---------------------------------------------------------
# Initialize Basic Scene
#---------------------------------------------------------

engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)

scene_config = sapien.SceneConfig()
scene = engine.create_scene(scene_config)
timestep = 1 / 100.0
scene.set_timestep(timestep)
scene.add_ground(-1)
physical_material = scene.create_physical_material(static_friction=1, dynamic_friction=1, restitution=0.0)
scene.default_physical_material = physical_material


scene.set_ambient_light(color=[0.5, 0.5, 0.5])
scene.add_directional_light(direction=[0, 1, -1], color=[0.5, 0.5, 0.5], shadow=True)
scene.add_point_light(position=[1, 2, 2], color=[1, 1, 1], shadow=True)
scene.add_point_light(position=[1, -2, 2], color=[1, 1, 1], shadow=True)
scene.add_point_light(position=[-1, 0, 1], color=[1, 1, 1], shadow=True)

viewer = Viewer(renderer)
viewer.set_scene(scene)
viewer.set_camera_xyz(x=1.46, y=0, z=0.6)
viewer.set_camera_rpy(r=0, p=-0.8, y=np.pi)
viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

#---------------------------------------------------------
# Create Tabletop
#---------------------------------------------------------

# table top
table=create_table(
    scene=scene,
    pose=sapien.Pose([0.56, 0, 0]),
    size=1.0,
    height=1,
    thickness=0.1,
    name="table",
    )

#objects
box = create_box(
    scene,
    sapien.Pose(p=[0.56, 0, 0.02], q=euler2quat(0, 0, np.pi/2)),
    half_size=[0.02, 0.05, 0.02],
    color=[1., 0., 0.],
    name='box',
)
sphere = create_sphere(
    scene,
    sapien.Pose(p=[-0.3+0.56, -0.4, 0.02]),
    radius=0.02,
    color=[0., 1., 0.],
    name='sphere',
)
capsule = create_capsule(
    scene,
    sapien.Pose(p=[0.3+0.3, 0.2, 0.02]),
    radius=0.02,
    half_length=0.03,
    color=[0., 0., 1.],
    name='capsule',
)
banana = load_object_mesh(
    scene, 
    sapien.Pose(p=[-0.2+0.56, 0, 0.01865]), 
    collision_file_path=manipulate_root_path+'assets/object/banana/collision_meshes/collision.obj',
    visual_file_path=manipulate_root_path+'assets/object/banana/visual_meshes/visual.dae',
    name='banana',
)



#---------------------------------------------------------
# Hold on the GUI Window
#---------------------------------------------------------

while not viewer.closed:
    scene.update_render()
    viewer.render()