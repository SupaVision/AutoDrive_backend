from src.splaTAM.slam.optimizer import initialize_optimizer, get_loss
from src.splaTAM.slam.tracker import keyframe_selection_overlap
from src.splaTAM.structures.camera import initialize_camera_pose, initialize_first_timestep
from src.splaTAM.structures.gaussians import add_new_gaussians
from src.splaTAM.utils import save_params
from src.splaTAM.utils.geometry import build_rotation, prune_gaussians, densify, matrix_to_quaternion
from src.utils import get_dataset
from src.utils.dataset_loader import load_dataset_config
from .base import Evaluator, ModelTrainer, DatasetLoader
import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


class SLAMSystem:
    def __init__(self, config):
        self.config = config
        self.dataset_loader = DatasetLoader(config)
        self.model_trainer = ModelTrainer(config)
        self.evaluator = Evaluator(config)

    def run(self):
        data = self.dataset_loader.load_data()
        preprocessed_data = self.dataset_loader.preprocess_data(data)
        self.model_trainer.train()
        self.evaluator.evaluate()


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)


def rgbd_slam(config: dict):
    # return
    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
                dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False

    # TODO: not good
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
                dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False

    # TODO: important load,dataload
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)
        print(f"Setting num_frames to {num_frames}")

    # TODO: Initialize Parameters & Canoncial Camera parameters
    params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames,
                                                                                    config[
                                                                                        'scene_radius_depth_ratio'],
                                                                                    config['mean_sq_dist_method'],
                                                                                    gaussian_distribution=config[
                                                                                        'gaussian_distribution'])

    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []

    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []

    # TODO: main loop
    # Iterate over Scan
    for time_idx in tqdm(range(num_frames)):
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}

        tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']

        # Initialize the camera pose for the current frame
        # NOTE: constant speed model for init camera pose
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # NOTE: Tracking,what's use_gt_poses?
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # NOTE: init Optimizer          Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False

            # NOTE: get Loss for current frame
            loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx,
                                               config['tracking']['loss_weights'],
                                               config['tracking']['use_sil_for_loss'],
                                               config['tracking']['sil_thres'],
                                               config['tracking']['use_l1'],
                                               config['tracking']['ignore_outlier_depth_loss'], tracking=True,
                                               plot_dir=eval_dir, visualize_tracking_loss=config['tracking'][
                    'visualize_tracking_loss'],
                                               tracking_iteration=iter)

            # Backprop
            loss.backward()
            # Optimizer Update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                # Save the best candidate rotation & translation
                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                    candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()

            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        # NOTE: ground truth poses
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran

        # NOTE: Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx + 1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification

                densify_curr_data = curr_data

                # NOTE: Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data,
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], config['gaussian_distribution'])
            # NOTE: Keyframe-based Mapping
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size'] - 2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1],
                                                                num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list) - 1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # NOTE: init optimizer
            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)

            # Mapping

            # NOTE: keyframes
            # Randomly select a frame until current time step amongst keyframes
            rand_idx = np.random.randint(0, len(selected_keyframes))
            selected_rand_keyframe_idx = selected_keyframes[rand_idx]
            if selected_rand_keyframe_idx == -1:
                # Use Current Frame Data
                iter_time_idx = time_idx
                iter_color = color
                iter_depth = depth
            else:
                # Use Keyframe Data
                iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
            iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx + 1]
            iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx,
                         'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}

            # NOTE: get loss
            # Loss for current frame
            loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx,
                                               config['mapping']['loss_weights'],
                                               config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                               config['mapping']['use_l1'],
                                               config['mapping']['ignore_outlier_depth_loss'], mapping=True)
            # report_loss()here
            # Backprop
            loss.backward()
            with torch.no_grad():
                # NOTE Prune Gaussians
                if config['mapping']['prune_gaussians']:
                    params, variables = prune_gaussians(params, variables, optimizer, iter,
                                                        config['mapping']['pruning_dict'])
                # NOTE: Gaussian-Splatting's Gradient-based Densification
                if config['mapping']['use_gaussian_splatting_densification']:
                    params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])

                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # NOTE: Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx + 1) % config['keyframe_every'] == 0) or \
            (time_idx == num_frames - 2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (
                not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)

        torch.cuda.empty_cache()

    # NOTE: save results
    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)

    # Save Parameters
    save_params(params, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    # TODO: set Experiment Seed
    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])

    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))
    # print(f"config: {experiment.config}")
    rgbd_slam(experiment.config)
