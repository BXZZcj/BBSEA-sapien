import sapien.core as sapien
from sapien.core import Scene, ActorBase, Actor, RenderBody, Articulation, Link
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
import open3d as o3d
from typing import Union, Tuple

from scene.core import SpecifiedObject

def get_pose_by_name(
        scene:Scene,
        name:str,
) -> sapien.Pose:
    for actor in scene.get_all_actors():
        if name==actor.get_name():
            return actor.get_pose()
    return None


def get_actor_by_name(
        scene:Scene,
        name:str,
) -> ActorBase:
    for actor in scene.get_all_actors():
        if name==actor.get_name():
            return actor
    return None
        

def get_articulation_by_name(
        scene:Scene,
        name:str,
) -> Articulation:
    for articulation in scene.get_all_articulations():
        if name==articulation.get_name():
            return articulation
    return None


def get_actor_names_in_scene(
        scene: Scene,
) -> list:
    name_list=[]
    for actor in scene.get_all_actors():
        name_list.append(actor.get_name())
    return name_list


def get_articulation_names_in_scene(
        scene: Scene,
) -> list:
    name_list=[]
    for articulation in scene.get_all_articulations():
        name_list.append(articulation.get_name())
    return name_list


def get_object_by_name(
        scene:Scene,
        name:str,
) -> ActorBase:
    for actor in scene.get_all_actors():
        if name==actor.get_name():
            return actor
    return None


def get_pcd_from_obj(
        obj: Union[Actor,Link,Articulation,SpecifiedObject],
        dense_sample_convex:bool=False
)->np.ndarray:
    if type(obj)==Actor or type(obj)==Link:
        return get_pcd_from_actor(obj, dense_sample_convex)
    elif type(obj)==Articulation:
        return get_pcd_from_articulation(obj, dense_sample_convex)
    elif isinstance(obj, SpecifiedObject):
        return obj.get_pcd(dense_sample_convex)
    else:
        return None
        

def get_pcd_from_articulation(
        articulation: Articulation, 
        dense_sample_convex:bool=False
) -> np.ndarray:
    pcd=[]
    for link in articulation.get_links():
        for vis_body in link.get_collision_visual_bodies():
            for render_shape in vis_body.get_render_shapes():
                vertices = render_shape.mesh.vertices
                if dense_sample_convex:
                    vertices=_dense_sample_convex_pcd(vertices)

                # import open3d as o3d
                # pcd_ = o3d.geometry.PointCloud()
                # pcd_.points = o3d.utility.Vector3dVector(vertices)
                # o3d.visualization.draw_geometries([pcd_], window_name="Open3D Point Cloud Visualization")

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


def get_pcd_from_actor(
        actor: Union[Actor, Link],
        dense_sample_convex:bool=False
) -> np.ndarray:
    pcd=np.array([]).reshape(0,3)
    for vis_body in actor.get_visual_bodies():
        part_pcd = _get_pcd_from_single_actor(actor, vis_body, dense_sample_convex)
        pcd=np.concatenate((pcd, part_pcd), axis=0)
    
    return pcd


def _get_pcd_from_single_actor(
        actor: Union[Actor, Link], 
        vis_body: RenderBody,
        dense_sample_convex:bool=False
) -> np.ndarray:    
    render_shape = vis_body.get_render_shapes()[0]
    vertices = render_shape.mesh.vertices
    # You must get the scale of the box, or the radius of the sphere, and in spaien 2.2.0, 
    # you can only get it from the actor builder. However, you cannot directly get Link (actor)
    # builder merely through actor.get_builder(). So we need special case handling, just as follows.
    if type(actor)==Actor:
        actor_builder = actor.get_builder()
    elif type(actor)==Link:
        for builder in actor.get_articulation().get_builder().get_link_builders():
            # if builder.get_name()==actor.get_name():
            if builder.get_index()==actor.get_index():
                actor_builder=builder
                break

    if vis_body.type == "box":
        vertices = _dense_sample_convex_pcd(vertices * actor_builder.get_visuals()[0].scale)
    elif vis_body.type == "sphere":
        vertices = vertices * actor_builder.get_visuals()[0].radius
    elif vis_body.type == "capsule":
        vertices = vertices
    elif vis_body.type == "mesh":
        vertices = vertices * vis_body.scale
    
    if dense_sample_convex:
        vertices=_dense_sample_convex_pcd(vertices)

    tf_mat = actor.get_pose().to_transformation_matrix() @ vis_body.local_pose.to_transformation_matrix()
    vertices_homo = np.concatenate((vertices, np.ones((vertices.shape[0],1))), axis=-1)
    pcd = (tf_mat @ vertices_homo.T).T[:,:-1]
    
    return pcd


def _dense_sample_convex_pcd(point_cloud: np.ndarray) -> np.ndarray:
    sample_count = 10000
    if len(point_cloud) > sample_count:
        return point_cloud
    
    # In case the pointcloud cannot be a convex for dense sampling
    normal_vector, are_coplanar = _are_points_coplanar(point_cloud)
    if are_coplanar:
        point_cloud_upward=point_cloud + normal_vector*0.000001
        point_cloud_downward=point_cloud - normal_vector*0.000001
        point_cloud=np.concatenate((point_cloud_upward, point_cloud_downward), axis=0)
    
    hull = ConvexHull(point_cloud)
    mesh = trimesh.Trimesh(vertices=point_cloud, faces=hull.simplices)
    denser_pcd, _ = trimesh.sample.sample_surface(mesh, sample_count)

    return np.array(denser_pcd)


def _are_points_coplanar(points:np.ndarray)->Union[np.ndarray, bool]:
    if len(points) < 4:
        return None, True 
    
    p1, p2, p3 = points[:3]
    
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    
    normal_vector = np.cross(v1, v2)
    normal_vector=normal_vector/np.linalg.norm(normal_vector)
    
    for point in points[3:]:
        v = np.array(point) - np.array(p1)
        if not np.isclose(np.dot(normal_vector, v), 0, atol=1e-8):
            return None, False
    
    return normal_vector, True


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


def get_pcd_normals_from_obj(
        obj: Union[Actor, Articulation, SpecifiedObject],
        dense_sample_convex:bool=False
)->Tuple[np.ndarray, np.ndarray]:
    if type(obj)==Actor:
        return get_pcd_normals_from_actor(obj, dense_sample_convex)
    elif type(obj)==Articulation:
        return get_pcd_normals_from_articulation(obj, dense_sample_convex)
    elif isinstance(obj, SpecifiedObject):
        return obj.get_pcd_normals(dense_sample_convex)
    else:
        return None


def get_pcd_normals_from_actor(
        actor: Actor,
        dense_sample_convex:bool=False
) -> Tuple[np.ndarray, np.ndarray]:
    # You can directly get normals from actors which are loaded from mesh files
    if actor.get_builder().get_visuals()[0].type == "File":
        return get_pcd_from_actor(actor), actor.get_visual_bodies()[0].get_render_shapes()[0].mesh.normals
    
    pcd = np.array([]).reshape(0, 3)
    normals = np.array([]).reshape(0, 3)
    for vis_body in actor.get_visual_bodies():
        part_pcd = _get_pcd_from_single_actor(actor, vis_body, dense_sample_convex)
        part_normals = _get_normals_from_single_actor(part_pcd)
        pcd = np.concatenate((pcd, part_pcd), axis=0)
        normals = np.concatenate((normals, part_normals), axis=0)

    return pcd, normals


def get_pcd_normals_from_articulation(
        articulation: Articulation,
        dense_sample_convex:bool=False        
) -> np.ndarray:
    pcd=np.array([]).reshape(0, 3)
    normals=np.array([]).reshape(0, 3)
    for link in articulation.get_links():
        part_pcd, part_normals = get_pcd_normals_from_actor(link, dense_sample_convex)
        pcd=np.concatenate((pcd, part_pcd), axis=0)
        normals=np.concatenate((normals, part_normals), axis=0)

    # If the articulation.get_links()[1].get_visual_bodies() return [], just change the index 1 to other number
    pcd=pcd * articulation.get_links()[1].get_visual_bodies()[0].scale

    tf_mat=articulation.get_links()[1].get_pose().to_transformation_matrix()
    pcd_homo=np.concatenate((pcd, np.ones((pcd.shape[0],1))), axis=-1)
    pcd = (tf_mat@pcd_homo.T).T[:,:-1]

    # hull = ConvexHull(pcd)
    # pcd = pcd[hull.vertices, :]
    
    return pcd, normals


def _get_normals_from_single_actor(point_cloud: np.ndarray) -> np.ndarray:
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