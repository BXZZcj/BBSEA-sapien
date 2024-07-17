import sapien.core as sapien
from sapien.core import ActorBase, Actor
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
import open3d as o3d

def get_pose_by_name(
        scene:sapien.Scene,
        name:str,
) -> sapien.Pose:
    for actor in scene.get_all_actors():
        if name==actor.get_name():
            return actor.get_pose()


def get_actor_by_name(
        scene:sapien.Scene,
        name:str,
) -> sapien.ActorBase:
    for actor in scene.get_all_actors():
        if name==actor.get_name():
            return actor


def get_names_in_scene(
        scene: sapien.Scene,
) -> list:
    name_list=[]
    for actor in scene.get_all_actors():
        name_list.append(actor.get_name())
    return name_list

def get_object_by_name(
        scene:sapien.Scene,
        name:str,
) -> ActorBase:
    for actor in scene.get_all_actors():
        if name==actor.get_name():
            return actor
        

def get_pcd_from_actor(actor: Actor) -> np.ndarray:
    vis_body = actor.get_visual_bodies()[0]
    render_shape = vis_body.get_render_shapes()[0]
    vertices = render_shape.mesh.vertices

    actor_type = actor.get_builder().get_visuals()[0].type
    if actor_type == "Box":
        vertices = vertices * actor.get_builder().get_visuals()[0].scale 
    elif actor_type == "Sphere":
        vertices = vertices * actor.get_builder().get_visuals()[0].radius
    elif actor_type == "Capsule" or actor_type == "File":
        vertices = vertices

    tf_mat=actor.get_pose().to_transformation_matrix()
    vertices_homo=np.concatenate((vertices, np.ones((vertices.shape[0],1))), axis=-1)
    pcd = (tf_mat@vertices_homo.T).T[:,:-1]
    
    return pcd


def dense_sample_pcd(point_cloud: np.ndarray) -> np.ndarray:
    hull = ConvexHull(point_cloud)
    mesh = trimesh.Trimesh(vertices=point_cloud, faces=hull.simplices)
    denser_pcd, _ = trimesh.sample.sample_surface(mesh, 10000)

    return denser_pcd


def get_normals_from_actor(actor: Actor, point_cloud: np.ndarray) -> np.ndarray:
    # You can directly get normals from actors which are loaded from mesh files
    if actor.get_builder().get_visuals()[0].type == "File":
        return actor.get_visual_bodies()[0].get_render_shapes()[0].mesh.normals
    actor.get_visual_bodies()[0].get_render_shapes()[0].mesh.vertices

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals()
    
    # We assume the actors created by sapien are convex
    centroid = np.mean(point_cloud, axis=0)
    for i, point in enumerate(point_cloud):
        normal = np.asarray(pcd.normals)[i]        
        vector_to_centroid = centroid - point        
        if np.dot(normal, vector_to_centroid) > 0:
            pcd.normals[i] = -pcd.normals[i]
    
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    normals = np.asarray(pcd.normals)
    
    return normals