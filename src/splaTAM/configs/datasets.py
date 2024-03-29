from pathlib import Path

import cv2
import numpy as np
import torch
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset

from src.splaTAM.structures.RGB_D_images import RGBDImage
from src.splaTAM.utils.geometry import as_intrinsics_matrix, relative_transformation


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


class BaseDataset(Dataset):
    """
    :var scale: tuple[int, int] desired width and height of the images
    """

    def __init__(
        self,
        cfg: dict,
        input_folder: Path,
        scale: float,
        device: str = "cuda:0",
        relative_pose: bool = False,
    ):
        super().__init__()
        self.name = cfg["dataset"]
        self.input_folder = input_folder
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg["cam"]["png_depth_scale"]

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            cfg["cam"]["H"],
            cfg["cam"]["W"],
            cfg["cam"]["fx"],
            cfg["cam"]["fy"],
            cfg["cam"]["cx"],
            cfg["cam"]["cy"],
        )

        self.distortion = (
            np.array(cfg["cam"]["distortion"]) if "distortion" in cfg["cam"] else None
        )
        # self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None
        #
        # self.crop_edge = cfg['cam']['crop_edge']

        self.color_paths, self.depth_paths = self.filepaths()
        self.num_img = len(self.color_paths)
        self.poses = self.load_poses()
        self.poses = torch.stack(self.poses)
        if relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_img

    def __getitem__(self, index) -> RGBDImage:
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = cv2.imread(color_path.as_posix())
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        if depth_path.suffix == ".png":
            depth = cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED)
        # elif '.exr' in depth_path:
        #     depth_data = readEXR_onlydepth(depth_path)
        else:
            raise ValueError(f"Unsupported depth file format {depth_path.suffix}.")

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        color = torch.from_numpy(color)
        depth = torch.from_numpy(depth)
        K = torch.from_numpy(K)
        rgb_d_image = RGBDImage(
            color, depth, K, self.poses[index], device=self.device, scale=self.scale
        )

        # edge = self.crop_edge
        # if edge > 0:
        #     # crop image edge, there are invalid value on the edge of the color image
        #     color = color[edge:-edge, edge:-edge]
        #     depth = depth[edge:-edge, edge:-edge]
        # pose = self.poses[index]
        # pose[:3, 3] *= self.scale

        return rgb_d_image

    def _preprocess_poses(self, poses: torch.Tensor) -> Tensor:
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogeneous transformation for all other poses.
        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed
        Returns:
            Output (torch.Tensor): Preprocessed poses
        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )

    def load_poses(self) -> list[Tensor]:
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def filepaths(self) -> tuple[list[Path], list[Path]]:
        """
        :return: rgp and depth file paths
        """
        raise NotImplementedError


class Replica(BaseDataset):
    def __init__(
        self,
        cfg: dict,
        input_folder: Path,
        scale: float,
        device: str = "cuda:0",
        relative_pose: bool = False,
    ):
        super().__init__(cfg, input_folder, scale, device, relative_pose)

    def load_poses(self) -> list[Tensor]:
        poses = []
        pose_path = self.input_folder / "traj.txt"
        with open(pose_path) as f:
            lines = f.readlines()
        for i in range(self.num_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def filepaths(self) -> tuple[list[Path], list[Path]]:
        color_paths = natsorted(self.input_folder.rglob("frame*.jpg"))
        depth_paths = natsorted(self.input_folder.rglob("depth*.png"))
        if len(color_paths) == 0 or len(depth_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.input_folder}. Please check the path."
            )
        elif len(color_paths) != len(depth_paths):
            raise ValueError(
                f"Number of color and depth images do not match in {self.input_folder}."
            )
        return color_paths, depth_paths
