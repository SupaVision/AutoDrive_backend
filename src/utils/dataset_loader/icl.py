import glob
import os
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class ICLDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = 1,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 480,
        desired_width: int | None = 640,
        load_embeddings: bool | None = False,
        embedding_dir: Path | str | None = "embeddings",
        embedding_dim: int | None = 512,
        embedding_file_extension: str | None = "pt",
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # Attempt to find pose file (*.gt.sim)
        self.pose_path = glob.glob(os.path.join(self.input_folder, "*.gt.sim"))
        if self.pose_path == 0:
            raise ValueError("Need pose file ending in extension `*.gt.sim`")
        self.pose_path = self.pose_path[0]
        self.embedding_file_extension = embedding_file_extension
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(
                    f"{self.input_folder}/{self.embedding_dir}/*.{self.embedding_file_extension}"
                )
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []

        lines = []
        with open(self.pose_path) as f:
            lines = f.readlines()

        _posearr = []
        for line in lines:
            line = line.strip().split()
            if len(line) == 0:
                continue
            _npvec = np.asarray(
                [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            )
            _posearr.append(_npvec)
        _posearr = np.stack(_posearr)

        for pose_line_idx in range(0, _posearr.shape[0], 3):
            _curpose = np.zeros((4, 4))
            _curpose[3, 3] = 3
            _curpose[0] = _posearr[pose_line_idx]
            _curpose[1] = _posearr[pose_line_idx + 1]
            _curpose[2] = _posearr[pose_line_idx + 2]
            poses.append(torch.from_numpy(_curpose).float())

        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
