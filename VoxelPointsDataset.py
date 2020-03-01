import numpy
import nrrd
import random
import os.path as osp
from config import cfg
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms
import torch


def RandomSplit(trainProportion=cfg.trainProportion, dataDir=cfg.dataDir):
    trainValList = list(range(1, cfg.dataSize + 1, 1))
    trainList = random.sample(trainValList,
                              int(len(trainValList) * trainProportion))
    trainList.sort()
    valList = [x for x in trainValList if x not in trainList]
    with open(osp.join(dataDir, 'train.csv'), 'w') as trainCsv:
        for item in trainList:
            trainCsv.write(str(item) + '\n')
    with open(osp.join(dataDir, 'val.csv'), 'w') as valCsv:
        for item in valList:
            valCsv.write(str(item) + '\n')


class VoxelPointsDataset(Dataset.Dataset):
    def __init__(self, dataDir=cfg.dataDir, isTrain=True):
        self.dataDir = dataDir
        self.nameList = []
        self.size = 0
        self.transform = transforms.ToTensor()

        strFileName = ""
        if isTrain is True:
            strFileName = "train.csv"
        else:
            strFileName = "val.csv"

        with open(osp.join(dataDir, strFileName)) as file:
            for f in file:
                self.nameList.append(f.strip("\n"))
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        nrrdPath = osp.join(self.dataDir, "NrrdData",
                            self.nameList[idx] + ".nrrd")
        nrrdData, header = nrrd.read(filename=nrrdPath)
        voxel = self.transform(nrrdData)
        voxel = voxel.view(1, 32, 32, 32)

        # points cloud # unfinished
        points = list([(0., 0., 0.), (1., 1., 1.)])
        points = torch.tensor(points)

        # check out the clostest point of each voxel # unfinished
        nearestPointOfVoxel = torch.zeros(1, 32, 32, 32)

        sample = {
            'voxel': voxel,
            'points': points,
            'nearestPointOfVoxel': nearestPointOfVoxel
        }
        return sample


if __name__ == '__main__':
    pass
