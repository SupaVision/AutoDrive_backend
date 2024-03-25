import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from src.splaTAM.structures.camera import get_pointcloud
from src.splaTAM.utils.geometry import (
    build_rotation,
    transform_to_frame,
    transformed_params2depthplussilhouette,
)


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        "means3D": means3D,
        "rgb_colors": new_pt_cld[:, 3:6],
        "unnorm_rotations": unnorm_rots,
        "logit_opacities": logit_opacities,
        "log_scales": log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(
                torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
            )
        else:
            params[k] = torch.nn.Parameter(
                v.cuda().float().contiguous().requires_grad_(True)
            )

    return params


def add_new_gaussians(
    params,
    variables,
    curr_data,
    sil_thres,
    time_idx,
    mean_sq_dist_method,
    gaussian_distribution,
):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(
        params, time_idx, gaussians_grad=False, camera_grad=False
    )
    depth_sil_rendervar = transformed_params2depthplussilhouette(
        params, curr_data["w2c"], transformed_gaussians
    )
    (
        depth_sil,
        _,
        _,
    ) = Renderer(
        raster_settings=curr_data["cam"]
    )(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = silhouette < sil_thres
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data["depth"][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (
        depth_error > 50 * depth_error.median()
    )
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(
            params["cam_unnorm_rots"][..., time_idx].detach()
        )
        curr_cam_tran = params["cam_trans"][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = curr_data["depth"][0, :, :] > 0
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(
            curr_data["im"],
            curr_data["depth"],
            curr_data["intrinsics"],
            curr_w2c,
            mask=non_presence_mask,
            compute_mean_sq_dist=True,
            mean_sq_dist_method=mean_sq_dist_method,
        )
        new_params = initialize_new_params(
            new_pt_cld, mean3_sq_dist, gaussian_distribution
        )
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(
                torch.cat((params[k], v), dim=0).requires_grad_(True)
            )
        num_pts = params["means3D"].shape[0]
        variables["means2D_gradient_accum"] = torch.zeros(
            num_pts, device="cuda"
        ).float()
        variables["denom"] = torch.zeros(num_pts, device="cuda").float()
        variables["max_2D_radius"] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx * torch.ones(new_pt_cld.shape[0], device="cuda").float()
        variables["timestep"] = torch.cat((variables["timestep"], new_timestep), dim=0)

    return params, variables
