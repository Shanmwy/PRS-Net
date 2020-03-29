import torch
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import nrrd
import os.path as osp
from config import cfg
import random

for idx in range(1, 97, 1):
    # read nrrd voxel data
    nrrdPath = osp.join(cfg.dataDir, 'MatureData', str(idx), "model.nrrd")
    nrrdData, header = nrrd.read(filename=nrrdPath)
    transform = transforms.ToTensor()
    voxel = transform(nrrdData)
    voxel = voxel.view(1, 32, 32, 32)

    # read pcd points data
    pcdPath = osp.join(cfg.dataDir, 'MatureData', str(idx), "model.pcd")
    with open(pcdPath, mode='r') as pcdFile:
        for i in range(9):
            pcdFile.readline()
        line = pcdFile.readline()
        numPoints = line.split(sep=' ')[1]
        pcdFile.readline()
        points = list()
        for i in range(int(numPoints)):
            line = pcdFile.readline()
            xyz = line.split(' ')
            points.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

    # 补齐1000个点, 对齐点云与体素数据
    addition = random.sample(points, 1000 - int(numPoints))
    points.extend(addition)
    points = torch.tensor(points)
    max, _ = torch.max(points, dim=0)
    min, _ = torch.min(points, dim=0)
    maxDiff = torch.max(max - min)
    points = points - min
    points = points / maxDiff * 32 - 0.5

    # 绘图
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(0, 45)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='y')
    ax.voxels(voxel[0], color='w', edgecolor="b")
    figPath = osp.join(cfg.dataDir, 'MatureData', str(idx), "model.png")
    fig.savefig(figPath)

    # 重新输出点云
    updatedPcdPath = osp.join(cfg.dataDir, 'MatureData', str(idx),
                              "model.updatedpcd")
    with open(updatedPcdPath, mode='w') as pcdFile:
        for i in range(1000):
            pcdFile.write('{} {} {}\n'.format(float(points[i][0]),
                                              float(points[i][1]),
                                              float(points[i][2])))

    # 寻找每个体素中心的最近点
    nearestPointOfVoxel = torch.zeros(1, 32, 32, 32)
    for i in range(32):
        for j in range(32):
            for k in range(32):
                voxelPos = torch.tensor([i, j, k])
                distance = torch.norm(points - voxelPos, dim=1, keepdim=True)
                _, index = torch.min(distance, dim=0)
                nearestPointOfVoxel[0, i, j, k] = index

    # 输出最近点文件
    npvPath = osp.join(cfg.dataDir, 'MatureData', str(idx), "model.npv")
    with open(npvPath, mode='w') as npvFile:
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    npvFile.write('{} '.format(
                        int(nearestPointOfVoxel[0, i, j, k])))

    print('{}th sample is prepared'.format(idx))
