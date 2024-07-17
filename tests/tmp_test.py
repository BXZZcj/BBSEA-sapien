import open3d as o3d
import numpy as np

# 假设 points 是你的 N*3 的 numpy.ndarray 点云数据
points = np.random.rand(100, 3)  # 这里是生成一个示例点云

# 将 numpy 数组转换为 Open3D 的点云格式
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 计算法向量，这里使用的是默认参数
pcd.estimate_normals()

# 将法向量转换回 numpy 数组
normals = np.asarray(pcd.normals)

# 打印结果查看
print("Normals:\n", normals)
print("Shape of normals array:", normals.shape)
