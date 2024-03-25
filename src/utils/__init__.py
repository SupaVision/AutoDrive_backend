from __future__ import annotations

import logging
from pathlib import Path

import open3d as o3d

from .dataset_loader import (
    Ai2thorDataset,
    AzureKinectDataset,
    ICLDataset,
    NeRFCaptureDataset,
    RealsenseDataset,
    Record3DDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    ScannetDataset,
    ScannetPPDataset,
    TUMDataset,
)


def ply_show(file_path: Path | str):
    file_path = Path(file_path)
    logging.info(f"ply_show file_path: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path.as_posix(), print_progress=True)
    o3d.visualization.draw_geometries([pcd])


def load_and_show_ply(ply_file_path: Path | str):
    ply_file_path = Path(ply_file_path)
    # 加载PLY文件
    mesh = o3d.io.read_triangle_mesh(ply_file_path.as_posix())

    # 如果PLY有颜色信息，确保它被视为顶点颜色
    if mesh.has_vertex_colors():
        print("PLY file has color information.")

    # 显示模型
    o3d.visualization.draw_geometries([mesh])


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


if __name__ == "__main__":
    # ply_show("/home/atticuszz/DevSpace/python/AutoDrive_backend/Datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL_semantic.ply")
    load_and_show_ply(
        "/home/atticuszz/DevSpace/python/AutoDrive_backend/Datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL_semantic.ply"
    )
