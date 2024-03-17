#脚本下载

#wget http://kaldir.vc.in.tum.de/matterport/download_mp.py
#

#授予脚本文件执行权限

#chmod +x download_mp.py
#
#
##运行脚本文件下载数据集
# 搭建python2.7的虚拟环境
#
#conda create -n mp3d python=2.7
#
#conda activate mp3d

# python download_mp.py --task habitat -o /home/atticuszz/DevSpace/python/AutoDrive_backend/Datasets
# https://blog.csdn.net/qq_44100524/article/details/133967011
wget -c http://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip -O /home/atticuszz/DevSpace/python/AutoDrive_backend/Datasets/mp3d_habitat.zip
