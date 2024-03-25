from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class BaseConfig:
    # 共通的配置项
    workdir: str
    scene_name: str
    seed: int = 0  # 随机种子
    primary_device: str = "cuda:0"
    map_every: int = 1  # 每n帧进行一次映射
    keyframe_every: int = 5  # 每n帧设置一个关键帧
    mapping_window_size: int = 24  # 映射窗口大小
    report_global_progress_every: int = 500  # 每n帧报告一次全局进度
    eval_every: int = 5  # 每n帧进行一次评估
    scene_radius_depth_ratio: int = 3  # 场景半径深度比率
    mean_sq_dist_method: str = "projective"  # 平均平方距离计算方法
    gaussian_distribution: str = "isotropic"  # 高斯分布类型
    report_iter_progress: bool = False  # 是否报告迭代进度
    load_checkpoint: bool = False  # 是否加载检查点
    checkpoint_time_idx: int = 0  # 检查点时间索引
    save_checkpoints: bool = False  # 是否保存检查点
    checkpoint_interval: int = 100  # 检查点间隔
    use_wandb: bool = True  # 是否使用WandB

    @property
    def run_name(self):
        return f"{self.scene_name}_{self.seed}"


@dataclass
class WandbConfig:
    name: str
    group: str
    save_qual: bool = False  # 是否保存质量评估结果
    eval_save_qual: bool = True  # 是否在评估时保存质量结果
    entity: str = "theairlab"  # WandB实体名, 根据您的WandB账户自定义
    project: str = "SplaTAM"  # WandB项目名, 根据项目需求自定义


@dataclass
class DataSetConfig:
    sequence: str  # 数据集序列名, 必须自定义以匹配所使用的数据集
    basedir: Path
    gradslam_data_cfg: Path  # GradSLAM数据配置文件路径
    desired_image_height: int
    desired_image_width: int
    start: int = 0  # 序列开始的帧编号
    end: int = -1  # 序列结束的帧编号, -1表示使用整个序列
    stride: int = 1  # 读取帧的步长
    num_frames: int = -1  # 读取的总帧数, -1表示读取整个序列
    ignore_bad = False
    use_train_split = True
    seperate_densification_res = False

    @property
    def densification_image_height(self):
        return self.desired_image_height

    @property
    def densification_image_width(self):
        return self.desired_image_width


@dataclass
class LossWeights:
    im: float = 0.5  # 图像损失权重
    depth: float = 1.0  # 深度损失权重


@dataclass
class LRates:
    means3D: float  # 3D均值的学习率
    rgb_colors: float  # RGB颜色的学习率
    unnorm_rotations: float  # 未标准化旋转的学习率
    logit_opacities: float  # 透明度logits的学习率
    log_scales: float  # 缩放比例的学习率
    cam_unnorm_rots: float  # 相机未标准化旋转的学习率
    cam_trans: float  # 相机平移的学习率


@dataclass
class TrackingConfig:
    lrs: LRates  # 学习率
    tracking_image_height: int
    tracking_image_width: int
    use_gt_poses: bool = False  # 是否使用地面真实姿态进行跟踪
    forward_prop: bool = True  # 是否前向传播姿态
    num_iters: int = 40  # 跟踪迭代次数
    use_sil_for_loss: bool = True  # 是否使用轮廓信息进行损失计算
    sil_thres: float = 0.99  # 轮廓阈值
    use_l1: bool = True  # 是否使用L1损失
    ignore_outlier_depth_loss: bool = False  # 是否忽略深度异常值
    loss_weights: LossWeights = field(default_factory=lambda: LossWeights())  # 损失权重
    use_depth_loss_thres = False
    depth_loss_thres = 100000
    visualize_tracking_loss = False
    seperate_tracking_res = False

    def set_image_size(self, height, width):
        self.tracking_image_height = height
        self.tracking_image_width = width


# visualize_tracking_loss=False, # Visualize Tracking Diff Images
# use_depth_loss_thres=True,
# depth_loss_thres=20000, # Num of Tracking Iters becomes twice if this value is not met
# use_uncertainty_for_loss_mask = False,
# use_uncertainty_for_loss = False,
# use_chamfer = False,


@dataclass
class PruningDict:
    start_after: int = 0  # 开始修剪的迭代次数
    remove_big_after: int = 0  # 移除大高斯体的迭代起始点
    stop_after: int = 20  # 停止修剪的迭代次数
    prune_every: int = 20  # 每隔多少迭代进行一次修剪
    removal_opacity_threshold: float = 0.005  # 移除高斯体的不透明度阈值
    final_removal_opacity_threshold: float = 0.005  # 最终移除的不透明度阈值
    reset_opacities: bool = False  # 是否重置不透明度
    reset_opacities_every: int = 500  # 每隔多少迭代重置不透明度


@dataclass
class DensifyDict:
    start_after: int = 500  # 开始增密的迭代次数
    remove_big_after: int = 3000  # 移除大高斯体的迭代起始点
    stop_after: int = 5000  # 停止增密的迭代次数
    densify_every: int = 100  # 每隔多少迭代进行一次增密
    grad_thresh: float = 0.0002  # 梯度阈值，用于决定是否增密
    num_to_split_into: int = 2  # 增密时分裂成的高斯体数量
    removal_opacity_threshold: float = 0.005  # 移除高斯体的不透明度阈值
    final_removal_opacity_threshold: float = 0.005  # 最终移除的不透明度阈值
    reset_opacities_every: int = 3000  # 每隔多少迭代重置不透明度


@dataclass
class MappingConfig:
    num_iters: int = 60  # 映射迭代次数
    add_new_gaussians: bool = True  # 是否添加新的高斯体
    sil_thres: float = 0.5  # 轮廓阈值，用于添加新高斯体
    use_l1: bool = True  # 是否使用L1损失
    use_sil_for_loss: bool = False  # 是否使用轮廓信息计算损失
    ignore_outlier_depth_loss: bool = False  # 是否忽略深度异常值
    loss_weights: LossWeights = field(default_factory=lambda: LossWeights())  # 损失权重
    lrs: LRates = field(
        default_factory=lambda: LRates(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        )
    )  # 学习率
    prune_gaussians: bool = True  # 是否在映射时修剪高斯体
    pruning_dict: PruningDict = field(
        default_factory=lambda: PruningDict()
    )  # 修剪高斯体的配置
    use_gaussian_splatting_densification: bool = False  # 是否使用高斯体增密
    densify_dict: DensifyDict = field(
        default_factory=lambda: DensifyDict()
    )  # 高斯体增密配置


# use_uncertainty_for_loss_mask = False,
# use_uncertainty_for_loss = False,
# use_chamfer = False,


@dataclass
class VisualizationConfig:
    render_mode: Literal["color", "depth", "centers"] = (
        "color"  # 渲染模式 ['color', 'depth', 'centers']
    )
    offset_first_viz_cam: bool = True  # 是否偏移第一视角相机
    show_sil: bool = False  # 是否显示轮廓而不是RGB
    visualize_cams: bool = True  # 是否可视化相机视锥和轨迹
    viz_w: int = 600  # 可视化宽度
    viz_h: int = 340  # 可视化高度
    viz_near: float = 0.01  # 可视化近平面
    viz_far: float = 100.0  # 可视化远平面
    view_scale: int = 2
    viz_fps: int = 5  # 在线重建可视化的帧率
    enter_interactive_post_online: bool = True
