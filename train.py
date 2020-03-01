from VoxelPointsDataset import RandomSplit, VoxelPointsDataset
import argparse
from torch.utils.data import DataLoader
import torch
from config import cfg
import PRSNet as PN
import os.path as osp

parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i',
                    '--max-epoch',
                    type=int,
                    nargs='?',
                    default=10,
                    help='max epoch')
parser.add_argument('-b',
                    '--single-batch-size',
                    type=int,
                    nargs='?',
                    default=5,
                    help='set batch size for each gpu')
parser.add_argument('-l',
                    '--lr',
                    type=float,
                    nargs='?',
                    default=0.001,
                    help='set learning rate')
parser.add_argument('-rs',
                    '--random-split',
                    action="store_true",
                    help='randomly split the dataset')
parser.add_argument('-w',
                    '--wr',
                    type=float,
                    nargs='?',
                    default=100,
                    help='weight of LossRegularization')
args = parser.parse_args()


def main():
    if args.random_split is True:
        RandomSplit(cfg.trainProportion, cfg.dataDir)

    trainSet = VoxelPointsDataset(cfg.dataDir, True)
    trainLoader = DataLoader(trainSet,
                             batch_size=args.single_batch_size,
                             shuffle=True)

    PRS_Net = PN.PRS_Net()
    LossSymmetryDistance = PN.LossSymmetryDistance()
    LossRegularization = PN.LossRegularization()

    optimizer = torch.optim.Adam(PRS_Net.parameters(), lr=args.lr)

    # training begins
    for epoch in range(args.max_epoch):
        for i, sample in enumerate(trainLoader, 0):
            voxel = sample['voxel']
            optimizer.zero_grad()
            outputs = PRS_Net(voxel)
            lsd = LossSymmetryDistance(outputs, sample)
            lreg = args.wr * LossRegularization(outputs)
            loss = lsd + lreg
            loss.backward()
            optimizer.step()

            print("{}th epoch, {}th sample, lsd is {}, lreg is {}".format(
                epoch, i, lsd, lreg))

    torch.save(PRS_Net, osp.join(cfg.modelDir, "PRS_Net.pkl"))
    print(outputs[0])


if __name__ == '__main__':
    main()