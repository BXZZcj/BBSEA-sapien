import requests
import json
import numpy as np
from typing import List, Dict, Union, Tuple
import open3d as o3d


def get_grasp_pose(
        env_pcd: np.ndarray,
        target_pcd: np.ndarray,
)->Tuple[List[Dict], List[Dict]]:
    # Make the input point cloud more compatible with the data feature which is adapted in the traing stage
    target_pcd_center = target_pcd.mean(axis=0)
    target_pcd = target_pcd[target_pcd[:, 2]>target_pcd_center[2]]

    rot_mat = np.array(
        [[1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]]
    )
    trans = np.array([-0.46, 0, -1])
    tf_mat=np.zeros(shape=(4, 4))
    tf_mat[:3, :3] = rot_mat
    tf_mat[:3, 3] = trans
    tf_mat[3, 3] = 1

    env_pcd_homo=np.concatenate((env_pcd, np.ones(shape=(env_pcd.shape[0], 1))), axis=-1)
    target_pcd_homo=np.concatenate((target_pcd, np.ones(shape=(target_pcd.shape[0], 1))), axis=-1)
    env_pcd = (tf_mat@env_pcd_homo.T).T[:,:-1]
    target_pcd = (tf_mat@target_pcd_homo.T).T[:,:-1]

    # Request the web api and parse the response
    graspnet_baseline_url = "http://localhost:6006/graspnet"

    env_pcd=env_pcd.tolist()
    target_pcd=target_pcd.tolist()
    data = {
        "target_pcd": target_pcd,
        "env_pcd": env_pcd,
    }
    
    json_data = json.dumps(data)
    response = requests.post(graspnet_baseline_url, data=json_data, headers={'Content-Type': 'application/json'})

    assert response.status_code == 200, "The remote server api requestion fails."
    
    response_json = response.json()
    grasp_poses = response_json['grasp_poses']
    serial_gripper_meshes = response_json['serial_gripper_meshes']

    # Reverse the data feature of the parsed response (including grasp poses and gripper meshes)
    # back to the origin.
    tf_mat_inv = np.linalg.inv(tf_mat)

    serial_gripper_vertices = np.array([mesh["vertices"] for mesh in serial_gripper_meshes])
    serial_gripper_vertices_homo = np.concatenate((serial_gripper_vertices, np.ones(shape=(serial_gripper_vertices.shape[0], serial_gripper_vertices.shape[1], 1))), axis=-1)
    reshaped_vertices_homo = serial_gripper_vertices_homo.reshape(-1, 4)
    serial_gripper_vertices = (tf_mat_inv @ reshaped_vertices_homo.T).T[:, :3].reshape(serial_gripper_vertices.shape[0], serial_gripper_vertices.shape[1], 3)

    for serial_gripper_mesh, serial_gripper_vert in zip(serial_gripper_meshes, serial_gripper_vertices):
        serial_gripper_mesh["vertices"] = serial_gripper_vert

    grasp_poses_translations = np.array([pose["translation"] for pose in grasp_poses])
    grasp_poses_translations_homo = np.concatenate((grasp_poses_translations, np.ones(shape=(grasp_poses_translations.shape[0], 1))), axis=-1)
    grasp_poses_translations = (tf_mat_inv @ grasp_poses_translations_homo.T).T[:, :3]

    grasp_poses_rotation_matrices = np.array([pose["rotation_matrix"] for pose in grasp_poses])
    grasp_poses_rotation_matrices = tf_mat_inv[:3, :3] @ grasp_poses_rotation_matrices

    for grasp_pose, grasp_poses_translation, grasp_poses_rotation_matrix in zip(grasp_poses, grasp_poses_translations, grasp_poses_rotation_matrices):
        grasp_pose["translation"] = grasp_poses_translation
        grasp_pose["rotation_matrix"] = grasp_poses_rotation_matrix

    # Convert the serial data of gripper meshes to o3d.geometry.TriangleMesh type
    gripper_meshes=[]
    for serial_gripper_mesh in serial_gripper_meshes:
        vertices = np.array(serial_gripper_mesh["vertices"])
        triangles = np.array(serial_gripper_mesh["faces"])
        vertex_colors = np.array(serial_gripper_mesh["vertex_colors"])

        gripper_mesh = o3d.geometry.TriangleMesh()
        gripper_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        gripper_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        gripper_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        gripper_meshes.append(gripper_mesh)

    return grasp_poses, gripper_meshes


def test_graspnet():
    graspnet_baseline_url = "http://localhost:6006/graspnet_demo"
    
    response = requests.post(graspnet_baseline_url, headers={'Content-Type': 'application/json'})

    assert response.status_code == 200, "The remote server api requestion fails."
    
    response_json = response.json()
    serial_cloud = response_json['serial_cloud']
    grasp_poses = response_json['grasp_poses']
    serial_gripper_meshes = response_json['serial_gripper_meshes']

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(serial_cloud["points"]).astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(np.array(serial_cloud["colors"]).astype(np.float32))

    gripper_meshes=[]
    for serial_gripper_mesh in serial_gripper_meshes:
        vertices = np.array(serial_gripper_mesh["vertices"])
        triangles = np.array(serial_gripper_mesh["faces"])
        vertex_colors = np.array(serial_gripper_mesh["vertex_colors"])

        gripper_mesh = o3d.geometry.TriangleMesh()
        gripper_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        gripper_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        gripper_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        gripper_meshes.append(gripper_mesh)

    return cloud, grasp_poses, gripper_meshes


if __name__=="__main__":
    cloud, grasp_poses, gripper_meshes = test_graspnet()
    o3d.visualization.draw_geometries([cloud, *gripper_meshes])

    env_pcd = np.random.rand(600000, 3)
    target_pcd = env_pcd[:50000]
    grasp_poses, gripper_meshes = get_grasp_pose(env_pcd, target_pcd)
        
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(env_pcd.astype(np.float32))

    print(grasp_poses)
    o3d.visualization.draw_geometries([cloud, *gripper_meshes])