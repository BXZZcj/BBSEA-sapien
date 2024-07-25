import sapien.core as sapien
from sapien.core import ActorBase, Actor, RenderBody
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
        

def get_articulation_by_name(
        scene:sapien.Scene,
        name:str,
) -> sapien.Articulation:
    for articulation in scene.get_all_articulations():
        if name==articulation.get_name():
            return articulation


def get_actor_names_in_scene(
        scene: sapien.Scene,
) -> list:
    name_list=[]
    for actor in scene.get_all_actors():
        name_list.append(actor.get_name())
    return name_list


def get_articulation_names_in_scene(
        scene: sapien.Scene,
) -> list:
    name_list=[]
    for articulation in scene.get_all_articulations():
        name_list.append(articulation.get_name())
    return name_list


def get_object_by_name(
        scene:sapien.Scene,
        name:str,
) -> ActorBase:
    for actor in scene.get_all_actors():
        if name==actor.get_name():
            return actor
        

def get_pcd_from_articulation(articulation: sapien.Articulation) -> np.ndarray:
    pcd=[]
    for link in articulation.get_links():
        for vis_body in link.get_visual_bodies():
            for render_shape in vis_body.get_render_shapes():
                vertices = render_shape.mesh.vertices
                pcd+=vertices.tolist()
    pcd=np.array(pcd)
    # If the articulation.get_links()[1].get_visual_bodies() return [], just change the index 1 to other number
    pcd=pcd * articulation.get_links()[1].get_visual_bodies()[0].scale

    tf_mat=articulation.get_links()[1].get_pose().to_transformation_matrix()
    pcd_homo=np.concatenate((pcd, np.ones((pcd.shape[0],1))), axis=-1)
    pcd = (tf_mat@pcd_homo.T).T[:,:-1]

    # hull = ConvexHull(pcd)
    # pcd = pcd[hull.vertices, :]
    
    return pcd


def get_pcd_from_actor(actor: sapien.Actor) -> np.ndarray:
    pcd=np.array([]).reshape(0,3)
    for vis_body in actor.get_visual_bodies():
        part_pcd = _get_pcd_from_actor(actor, vis_body)
        pcd=np.concatenate((pcd, part_pcd), axis=0)
    
    return pcd


def _get_pcd_from_actor(actor: Actor, vis_body: RenderBody) -> np.ndarray:
    render_shape = vis_body.get_render_shapes()[0]
    vertices = render_shape.mesh.vertices

    actor_type = actor.get_builder().get_visuals()[0].type
    if actor_type == "Box":
        vertices = _dense_sample_convex_pcd(vertices * actor.get_builder().get_visuals()[0].scale)
    elif actor_type == "Sphere":
        vertices = vertices * actor.get_builder().get_visuals()[0].radius
    elif actor_type == "Capsule" or actor_type == "File":
        vertices = vertices

    tf_mat = actor.get_pose().to_transformation_matrix() @ vis_body.local_pose.to_transformation_matrix()
    vertices_homo = np.concatenate((vertices, np.ones((vertices.shape[0],1))), axis=-1)
    pcd = (tf_mat @ vertices_homo.T).T[:,:-1]
    
    return pcd


def _dense_sample_convex_pcd(point_cloud: np.ndarray) -> np.ndarray:
    sample_count = 10000
    if len(point_cloud) > sample_count:
        return point_cloud
    
    hull = ConvexHull(point_cloud)
    mesh = trimesh.Trimesh(vertices=point_cloud, faces=hull.simplices)
    denser_pcd, _ = trimesh.sample.sample_surface(mesh, sample_count)

    return np.array(denser_pcd)


def uniform_sample_convex_pcd(point_cloud: np.ndarray) -> np.ndarray:
    sample_count = len(point_cloud)
    
    hull = ConvexHull(point_cloud)
    mesh = trimesh.Trimesh(vertices=point_cloud, faces=hull.simplices)
    uniform_pcd, _ = trimesh.sample.sample_surface(mesh, sample_count)

    return uniform_pcd


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