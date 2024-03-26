import logging

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["RGBDImage"]


def normalize_image(rgb: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    r"""Normalizes RGB image values from :math:`[0, 255]` range to :math:`[0, 1]` range.

    Args:
        rgb (torch.Tensor or numpy.ndarray): RGB image in range :math:`[0, 255]`

    Returns:
        torch.Tensor or numpy.ndarray: Normalized RGB image in range :math:`[0, 1]`

    Shape:
        - rgb: :math:`(*)` (any shape)
        - Output: Same shape as input :math:`(*)`
    """
    if torch.is_tensor(rgb):
        return rgb.float() / 255
    elif isinstance(rgb, np.ndarray):
        return rgb.astype(float) / 255
    else:
        raise TypeError(f"Unsupported input rgb type: {type(rgb)}")


def set_channels_first(rgb: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    r"""Converts from channels last representation :math:`(*, H, W, C)` to channels first representation
    :math:`(*, C, H, W)`

    Args:
        rgb (torch.Tensor or numpy.ndarray): :math:`(*, H, W, C)` ordering `(*, height, width, channels)`

    Returns:
        torch.Tensor or numpy.ndarray: :math:`(*, C, H, W)` ordering

    Shape:
        - rgb: :math:`(*, H, W, C)`
        - Output: :math:`(*, C, H, W)`
    """
    if not (isinstance(rgb, np.ndarray) or torch.is_tensor(rgb)):
        raise TypeError(f"Unsupported input rgb type {type(rgb)}")

    if rgb.ndim < 3:
        raise ValueError(
            f"Input rgb must contain at least 3 dims, but had {rgb.ndim} dims."
        )
    if rgb.shape[-3] < rgb.shape[-1]:
        logging.warning(
            f"Are you sure that the input is correct? Number of channels exceeds height of image: {rgb.shape[-1]} > {rgb.shape[-3]}"
        )
    ordering = list(range(rgb.ndim))
    ordering[-2], ordering[-1], ordering[-3] = ordering[-3], ordering[-2], ordering[-1]

    if isinstance(rgb, np.ndarray):
        return np.ascontiguousarray(rgb.transpose(*ordering))
    elif torch.is_tensor(rgb):
        return rgb.permute(*ordering).contiguous()


class RGBDImage:
    """
    Initialize an RGBDImage object.

    Args:
        rgb_image (Tensor): The RGB image tensor.
        depth_image (Tensor): The depth image tensor.
        intrinsics (Tensor): The camera intrinsics tensor. Typically, a 2D tensor [3, 3] for a single camera.
        pose (Tensor, optional): The camera pose tensor, a 4x4 transformation matrix.
        device (torch.device | str): The computation device ('cuda:0', 'cuda:1', 'cpu', etc.).
        size (tuple[int, int], optional): Target spatial size (height, width) for the images. If not provided, uses the depth image's size.
        scale (float): Scale factor to apply to the depth values.
        pixel_pos (Tensor, optional): The pixel positions tensor, if any.

    Raises:
        ValueError: If the input tensors do not meet the expected dimensional requirements.
    """

    def __init__(
        self,
        rgb_image: Tensor,
        depth_image: Tensor,
        intrinsics: Tensor,
        pose: Tensor | None = None,
        device: torch.device | str = "cuda:0",
        size: tuple[int, int] | None = None,
        scale: float = 1.0,
        *,
        pixel_pos: Tensor | None = None,
    ):
        # input ndim checks
        if intrinsics.ndim != 2:
            raise ValueError(
                f"intrinsics should have ndim=4, but had ndim={intrinsics.ndim}"
            )
        if pose is not None and pose.ndim != 2:
            raise ValueError(f"poses should have ndim=4, but had ndim={pose.ndim}")

        self._scale = scale
        self._device = device
        self._size = (
            size if size is not None else (depth_image.shape[0], depth_image.shape[1])
        )

        self._rgb_image = self._preprocess_color(rgb_image).to(self._device)
        self._depth_image = self._preprocess_depth(depth_image).to(self._device)
        self._intrinsics = intrinsics.to(self._device)

        if pose is not None:
            pose[:3, 3] *= self._scale
            self._pose = pose.to(self._device)
        else:
            self._pose = None
        self._pixel_pos = pixel_pos.to(self._device) if pixel_pos is not None else None

    def _preprocess_color(
        self,
        rgb_image: Tensor,
        *,
        normalize_color: bool = True,
        channels_first: bool = True,
        distortion: bool = False,
    ) -> Tensor:
        """
        Preprocesses the RGB image by resizing, optionally normalizing values to [0, 1],
        and optionally converting to channel-first format.

        Args:
            rgb_image (Tensor): The input RGB image tensor.
            normalize_color (bool): If True, normalize color values to [0, 1].
            channels_first (bool): If True, convert image to (C, H, W) format.
            distortion (bool): If True, apply distortion correction (not implemented).

        Returns:
            Tensor: The preprocessed RGB image tensor.
        """
        if rgb_image.ndim != 3:
            raise ValueError(
                f"rgb_image should have ndim=3, but had ndim={rgb_image.ndim}"
            )
        if channels_first:
            rgb_image = set_channels_first(rgb_image)

        # resize image
        rgb_image = F.interpolate(
            rgb_image.unsqueeze(0),
            size=self._size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if normalize_color:
            rgb_image = normalize_image(rgb_image)

        # if distortion:
        #     # un-distortion is only applied on color image, not depth!
        #     rgb_image = cv2.undistort(rgb_image, self._intrinsics, distortion)
        return rgb_image

    def _preprocess_depth(
        self, depth_image: Tensor, channels_first: bool = True
    ) -> Tensor:
        """
        Preprocesses the depth image by resizing and optionally converting to channel-first format.
        Scales the depth values according to the provided scale factor.

        Args:
            depth_image (Tensor): The raw depth image tensor.
            channels_first (bool): If True, convert depth to (1, H, W) format.

        Returns:
            Tensor: The preprocessed depth image tensor.
        """
        if depth_image.ndim != 2:
            raise ValueError(
                f"depth_image should have ndim=2, but had ndim={depth_image.ndim}"
            )
        if channels_first:
            depth_image = set_channels_first(depth_image)
        # resize depth image
        depth_image = (
            F.interpolate(
                depth_image.unsqueeze(0).unsqueeze(0), size=self._size, mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
        )

        return depth_image / self._scale
