from dataclasses import dataclass, field
from pathlib import Path

from .base import (
    BaseConfig,
    DataSetConfig,
    MappingConfig,
    VisualizationConfig,
    WandbConfig,
)

__all__ = ["ReplicaConfig"]
scenes = [
    "room0",
    "room1",
    "room2",
    "office0",
    "office1",
    "office2",
    "office_",
    "office4",
]
scene_name = scenes[0]
group_name = "replica"


@dataclass
class ReplicaConfig(BaseConfig):
    # 应用和实验的基本配置
    workdir: str = f"./experiments/{group_name}"  # 工作目录
    scene_name: str = scene_name
    mapping_window_size: int = 24  # 映射窗口大小
    report_global_progress_every: int = 500  # 每n帧报告一次全局进度
    checkpoint_time_idx: int = 0  # 检查点时间索引
    checkpoint_interval: int = 100  # 检查点间隔
    use_wandb: bool = True  # 是否使用WandB

    # 配置部分的实例化
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(name=scene_name, group=group_name)
    )  # 注意: 需要根据实际情况自定义'name'
    data: DataSetConfig = field(
        default_factory=lambda: DataSetConfig(
            sequence=scene_name,
            # TODO: fix the basedir
            basedir=Path(__file__).parents[5] / "Datasets" / group_name,
            gradslam_data_cfg=Path(__file__).parent
            / "gradslam_cfg"
            / f"{group_name}.yaml",
            desired_image_height=680,
            desired_image_width=1200,
            num_frames=10,
        )
    )
    # tracking: TrackingConfig = field(
    #     default_factory=lambda: TrackingConfig(num_iters=10, lrs=LRates(means3D=0.0,
    #                                                                     rgb_colors=0.0,
    #                                                                     unnorm_rotations=0.0,
    #                                                                     logit_opacities=0.0,
    #                                                                     log_scales=0.0,
    #                                                                     cam_unnorm_rots=0.0004,
    #                                                                     cam_trans=0.002)))
    mapping: MappingConfig = field(default_factory=MappingConfig)
    viz: VisualizationConfig = field(
        default_factory=lambda: VisualizationConfig(enter_interactive_post_online=True)
    )
