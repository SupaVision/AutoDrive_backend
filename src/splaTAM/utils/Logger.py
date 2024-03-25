import torch


def report_progress(
    params,
    tracking_curr_data,
    num_batches,
    progress_bar,
    iter_time_idx,
    sil_thres=0.5,
    tracking=False,
    wandb_run=None,
    wandb_step=None,
    wandb_save_qual=False,
    global_logging=False,
):
    """
    Report the progress of the current iteration.

    Args:
        params (dict): Dictionary containing all the parameters.
        tracking_curr_data (dict): Dictionary containing the current tracking data.
        num_batches (int): Number of batches.
        progress_bar (tqdm): Progress bar.
        iter_time_idx (int): Current time index.
        sil_thres (float): Silhouette threshold.
        tracking (bool): Whether to report tracking results.
        wandb_run (wandb.Run): Wandb run.
        wandb_step (int): Wandb step.
        wandb_save_qual (bool): Whether to save the qualitative results on wandb.
        global_logging (bool): Whether to log the results globally.

    Returns:
        None
    """
    # Get the current time index
    time_idx = iter_time_idx

    if time_idx == 0 or (time_idx + 1) % config["report_global_progress_every"] == 0:
        try:
            # Report Final Tracking Progress
            progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
            with torch.no_grad():
                if config["use_wandb"]:
                    report_progress(
                        params,
                        tracking_curr_data,
                        1,
                        progress_bar,
                        iter_time_idx,
                        sil_thres=config["tracking"]["sil_thres"],
                        tracking=True,
                        wandb_run=wandb_run,
                        wandb_step=wandb_time_step,
                        wandb_save_qual=config["wandb"]["save_qual"],
                        global_logging=True,
                    )
                else:
                    report_progress(
                        params,
                        tracking_curr_data,
                        1,
                        progress_bar,
                        iter_time_idx,
                        sil_thres=config["tracking"]["sil_thres"],
                        tracking=True,
                    )
            progress_bar.close()
        except:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            print("Failed to evaluate trajectory.")


def report_loss(losses, wandb_run, wandb_step, tracking=False, mapping=False):
    # Update loss dict
    loss_dict = {
        "Loss": losses["loss"].item(),
        "Image Loss": losses["im"].item(),
        "Depth Loss": losses["depth"].item(),
    }
    if tracking:
        tracking_loss_dict = {}
        for k, v in loss_dict.items():
            tracking_loss_dict[f"Per Iteration Tracking/{k}"] = v
        tracking_loss_dict["Per Iteration Tracking/step"] = wandb_step
        wandb_run.log(tracking_loss_dict)
    elif mapping:
        mapping_loss_dict = {}
        for k, v in loss_dict.items():
            mapping_loss_dict[f"Per Iteration Mapping/{k}"] = v
        mapping_loss_dict["Per Iteration Mapping/step"] = wandb_step
        wandb_run.log(mapping_loss_dict)
    else:
        frame_opt_loss_dict = {}
        for k, v in loss_dict.items():
            frame_opt_loss_dict[f"Per Iteration Current Frame Optimization/{k}"] = v
        frame_opt_loss_dict["Per Iteration Current Frame Optimization/step"] = (
            wandb_step
        )
        wandb_run.log(frame_opt_loss_dict)

    # Increment wandb step
    wandb_step += 1
    return wandb_step
