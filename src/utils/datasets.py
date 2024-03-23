import cv2
import torch
from natsort import natsorted
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from src.utils.common import as_intrinsics_matrix, set_channels_first, normalize_image


class BaseDataset(Dataset):
    """
    :var scale: tuple[int, int] desired height and width of the images
    """

    def __init__(self, cfg: dict, input_folder: Path, scale: tuple[int, int], device: str = 'cuda:0'
                 ):
        super().__init__()
        self.name = cfg['dataset']
        self.input_folder = input_folder
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder

        self.crop_edge = cfg['cam']['crop_edge']

        self.color_paths, self.depth_paths = self.filepaths()
        self.n_img = len(self.color_paths)

    def filepaths(self) -> tuple[list[Path], list[Path]]:
        """
        :return: rgp and depth file paths
        """
        jpg_dir: Path = self.input_folder / 'results'
        color_paths = natsorted([file for file in jpg_dir.glob('frame*.jpg')])
        depth_paths = natsorted([file for file in jpg_dir.glob('depth*.png')])
        if len(color_paths) == 0 or len(depth_paths) == 0:
            raise FileNotFoundError(
                f'No images found in {jpg_dir}. Please check the path.')
        elif len(color_paths) != len(depth_paths):
            raise ValueError(
                f'Number of color and depth images do not match in {jpg_dir}.')
        return color_paths, depth_paths

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path.as_posix())
        color_data = self._preprocess_color(color_data)
        if depth_path.suffix == '.png':
            depth_data = cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED)
        # elif '.exr' in depth_path:
        #     depth_data = readEXR_onlydepth(depth_path)
        else:
            raise ValueError(
                f'Unsupported depth file format {depth_path.suffix}.')

        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # un-distortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)
        # TODO: depth data is not checked
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape

        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data) * self.scale
        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        pose = self.poses[index]
        pose[:3, 3] *= self.scale
        return index, color_data.to(self.device), depth_data.to(self.device), pose.to(self.device)

    def _preprocess_color(self, color: np.ndarray, normalize_color: bool = True,
                          channels_first: bool = False) -> np.ndarray:
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.
        Args:
            color (np.ndarray): Raw input rgb image
        Reruns:
            np.ndarray: Preprocessed rgb image
        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        # weight and height
        color = cv2.resize(
            color,
            (self.scale[0], self.scale[1]),
            interpolation=cv2.INTER_LINEAR,
        )
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        if normalize_color:
            color = normalize_image(color)
        if channels_first:
            color = set_channels_first(color)
        return color
