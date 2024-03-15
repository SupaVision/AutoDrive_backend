import open3d as o3d

# 替换为你的点云文件路径
point_cloud_path = "/home/atticuszz/DevSpace/python/AutoDrive_backend/Datasets/cull_replica_mesh/office4.ply"
pcd = o3d.io.read_point_cloud(point_cloud_path, print_progress=True)

# 可视化点云
o3d.visualization.draw_geometries([pcd])


class Replica:
    pass


#  k
