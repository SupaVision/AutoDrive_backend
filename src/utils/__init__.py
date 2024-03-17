import logging
from pathlib import Path

import open3d as o3d



def ply_show(file_path:Path|str):
    file_path = Path(file_path)
    logging.info(f"ply_show file_path: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path.as_posix(), print_progress=True)
    o3d.visualization.draw_geometries([pcd])


def load_and_show_ply(ply_file_path:Path|str):
    ply_file_path = Path(ply_file_path)
    # 加载PLY文件
    mesh = o3d.io.read_triangle_mesh(ply_file_path.as_posix())

    # 如果PLY有颜色信息，确保它被视为顶点颜色
    if mesh.has_vertex_colors():
        print("PLY file has color information.")

    # 显示模型
    o3d.visualization.draw_geometries([mesh])

if __name__=="__main__":
    # ply_show("/home/atticuszz/DevSpace/python/AutoDrive_backend/Datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL_semantic.ply")
    load_and_show_ply("/home/atticuszz/DevSpace/python/AutoDrive_backend/Datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL_semantic.ply")
