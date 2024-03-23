
from gradslam import RGBDImages

from gradslam.slam import PointFusion
from src.utils import get_dataset
from src.utils.dataset_loader import load_dataset_config
from src.utils.dataset_loader.config import Replica

# 数据集路径
path_to_replica = '/path/to/replica'
cfg = load_dataset_config(Replica.data.gradslam_data_cfg.as_posix())
data_set = Replica.data
# 加载数据集
replica = get_dataset(cfg,basedir=data_set.basedir,sequence=data_set.sequence,)
colors, depths, intrinsics, poses, timestamps, filenames = replica[0]

# 根据相机参数调整内参
intrinsics.fx = 600.0
intrinsics.fy = 600.0
intrinsics.cx = 599.5
intrinsics.cy = 339.5

# 创建RGBD图像
rgbdimages = RGBDImages(colors, depths, intrinsics, poses, timestamps)
