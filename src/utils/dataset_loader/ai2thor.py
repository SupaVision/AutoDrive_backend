import glob
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted

from .basedataset import GradSLAMDataset


class Ai2thorDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 968,
        desired_width: int | None = 1296,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            if self.embedding_dir == "embed_semseg":
                # embed_semseg is stored as uint16 pngs
                embedding_paths = natsorted(
                    glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.png")
                )
            else:
                embedding_paths = natsorted(
                    glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
                )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        if self.embedding_dir == "embed_semseg":
            embedding = imageio.imread(embedding_file_path)  # (H, W)
            embedding = cv2.resize(
                embedding,
                (self.desired_width, self.desired_height),
                interpolation=cv2.INTER_NEAREST,
            )
            embedding = torch.from_numpy(embedding).long()  # (H, W)
            embedding = F.one_hot(
                embedding, num_classes=self.embedding_dim
            )  # (H, W, C)
            embedding = embedding.half()  # (H, W, C)
            embedding = embedding.permute(2, 0, 1)  # (C, H, W)
            embedding = embedding.unsqueeze(0)  # (1, C, H, W)
        else:
            embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
