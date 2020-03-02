from VoxelPointsDataset import VoxelPointsDataset
import argparse
from torch.utils.data import DataLoader
import torch
from config import cfg
import PRSNet as PN
import os.path as osp

parser = argparse.ArgumentParser(description='testing')
parser.add_argument(
    '-ml',
    '--minimal-lsd',
    type=float,
    nargs='?',
    default=0.0001,
    help=
    'the minimal lsd, planes or axes produce higher lsd than which would be eliminated'
)
parser.add_argument(
    '-mc',
    '--max-cos-dihedral-angle',
    type=float,
    nargs='?',
    default=0.866025,
    help=
    'the max cos dihedral angle, planes or axes have higher ones than which would be considered duplicated'
)
args = parser.parse_args()


def main():
    testSet = VoxelPointsDataset(cfg.dataDir, False)
    testLoader = DataLoader(testSet,
                            batch_size=1,
                            shuffle=True)

    PRS_Net = torch.load(osp.join(cfg.modelDir, "PRS_Net.pkl"))
    LossSymmetryDistance = PN.LossSymmetryDistance()
    LossRegularization = PN.LossRegularization()
    validateOutputs = PN.validateOutputs()

    with open(osp.join(cfg.resultDir, 'testLog.txt'), 'w') as testLog:
        # test begins
        for i, sample in enumerate(testLoader, 0):
            voxel = sample['voxel']
            outputs = PRS_Net(voxel)
            lsd = LossSymmetryDistance(outputs, sample)
            lreg = LossRegularization(outputs)
            outputs = outputs.view(6, 4)
            lsd = lsd.view(6)
            lreg = lreg.view(1)
            outputs = validateOutputs(outputs, lsd, args.minimal_lsd,
                                      args.max_cos_dihedral_angle)
            status = "{}th sample, lsd for each symmetry plane and rotation axis is {}, lreg is {}, validated outputs is {}\n".format(
                i, lsd, lreg, outputs)
            print(status)
            testLog.write(status)
        print('test finished')

if __name__ == '__main__':
    main()
