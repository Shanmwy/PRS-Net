from torch import nn
from config import cfg
import torch
import myQuaternion
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time


class PRS_Net(nn.Module):
    def __init__(self):
        super(PRS_Net, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(cfg.negativeSlope)
        self.ConvLayer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.ConvLayer2 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.ConvLayer3 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.ConvLayer4 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.ConvLayer5 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.FCLayerSP11 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerSP21 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerSP31 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerSP12 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerSP22 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerSP32 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerSP13 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)
        self.FCLayerSP23 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)
        self.FCLayerSP33 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)

        self.FCLayerRQ11 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerRQ21 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerRQ31 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerRQ12 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerRQ22 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerRQ32 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerRQ13 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)
        self.FCLayerRQ23 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)
        self.FCLayerRQ33 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)

    def forward(self, voxel):
        self.outputs = torch.zeros(voxel.shape[0], 6, 4)

        # print(torch.sum(voxel[0]-voxel[1]))
        voxel = self.ConvLayer1(voxel)
        # print(torch.sum(voxel[0]-voxel[1]))
        voxel = self.ConvLayer2(voxel)
        # print(torch.sum(voxel[0]-voxel[1]))
        voxel = self.ConvLayer3(voxel)
        # print(torch.sum(voxel[0]-voxel[1]))
        voxel = self.ConvLayer4(voxel)
        # print(torch.sum(voxel[0]-voxel[1]))
        voxel = self.ConvLayer5(voxel)  # voxel.shape = [batch_size,64,1,1,1]
        # print(torch.sum(voxel[0]-voxel[1]))
        voxel = voxel.view(voxel.shape[0], 64)
        # print(torch.sum(voxel[0]-voxel[1]))

        a = self.FCLayerSP11(voxel)
        a = self.FCLayerSP12(a)
        a = self.FCLayerSP13(a)
        self.assign2Outputs(self.unitize(a), 0)
        # print(a)

        a = self.FCLayerSP21(voxel)
        a = self.FCLayerSP22(a)
        a = self.FCLayerSP23(a)
        self.assign2Outputs(self.unitize(a), 1)

        a = self.FCLayerSP31(voxel)
        a = self.FCLayerSP32(a)
        a = self.FCLayerSP33(a)
        self.assign2Outputs(self.unitize(a), 2)

        a = self.FCLayerRQ11(voxel)
        a = self.FCLayerRQ12(a)
        a = self.FCLayerRQ13(a)
        self.assign2Outputs(self.unitize(a), 3)

        a = self.FCLayerRQ21(voxel)
        a = self.FCLayerRQ22(a)
        a = self.FCLayerRQ23(a)
        self.assign2Outputs(self.unitize(a), 4)

        a = self.FCLayerRQ31(voxel)
        a = self.FCLayerRQ32(a)
        a = self.FCLayerRQ33(a)
        self.assign2Outputs(self.unitize(a), 5)

        return self.outputs

    def unitize(self, a):
        return a / torch.norm(a, dim=1).view(-1, 1)

    def assign2Outputs(self, a: torch.tensor, index: int):
        for i in range(self.outputs.shape[0]):
            self.outputs[i][index] = a[i]
        return


class LossSymmetryDistance(object):
    def __call__(self, outputs: torch.tensor, sample):
        self.loss = torch.zeros(outputs.shape[0], 6)
        for i in range(outputs.shape[0]):
            self.voxel = sample['voxel'][i]
            self.points = sample['points'][i]
            self.nearestPointOfVoxel = sample['nearestPointOfVoxel'][i]

            for j in range(0, 3, 1):
                self.transformedPoints = self.reflectTransform(outputs[i][j])
                self.loss[i][j] = self.overAllDistance()

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.scatter(self.transformedPoints[:, 0].tolist(),
                           self.transformedPoints[:, 1].tolist(),
                           self.transformedPoints[:, 2].tolist(),
                           color='y')
                ax.scatter(self.points[:, 0],
                           self.points[:, 1],
                           self.points[:, 2],
                           color='b')
                fig.savefig(str(time.clock()) + '_' + str(j) + '.png')

            for j in range(3, 6, 1):
                self.transformedPoints = self.rotateTransform(outputs[i][j])
                self.loss[i][j] = 0  # self.overAllDistance()

        return self.loss

    def overAllDistance(self):
        sumD = 0
        for i in range(self.points.shape[0]):
            x = int(self.transformedPoints[i][0] + 0.5)
            y = int(self.transformedPoints[i][1] + 0.5)
            z = int(self.transformedPoints[i][2] + 0.5)
            x = 31 if x > 31 else x
            y = 31 if y > 31 else y
            z = 31 if z > 31 else z
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            z = 0 if z < 0 else z
            index = int(self.nearestPointOfVoxel[0, x, y, z])
            d = torch.norm(self.points[index] - self.transformedPoints[i])
            sumD += d
        return sumD

    def reflectTransform(self, planeParameters: torch.tensor):
        outPoints = torch.zeros_like(self.points)
        for i in range(self.points.shape[0]):
            outPoints[i] = self.points[i] - 2 * planeParameters[0:3] * (
                torch.dot(self.points[i], planeParameters[0:3]) +
                planeParameters[3]) / torch.dot(planeParameters[0:3],
                                                planeParameters[0:3])
        return outPoints

    def rotateTransform(self, q: torch.tensor):
        outPoints = torch.zeros_like(self.points)
        for i in range(self.points.shape[0]):
            outPoints[i] = myQuaternion.rotate(q, self.points[i])
        return outPoints


class LossRegularization(object):
    def __call__(self, outputs: torch.tensor):
        self.loss = torch.zeros(outputs.shape[0])

        for i in range(outputs.shape[0]):
            M1 = outputs[i][0:3, 0:3]
            M2 = outputs[i][3:6, 1:4]
            # print(outputs[i], M1, M2)
            M1 = M1 / torch.norm(M1, dim=1).view(-1, 1)
            M2 = M2 / torch.norm(M2, dim=1).view(-1, 1)
            II = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            A = torch.mm(M1, M1.t()) - II
            B = torch.mm(M2, M2.t()) - II
            self.loss[i] = torch.sum(A**2) + torch.sum(B**2)

        return self.loss


class validateOutputs(object):
    def __call__(self, outputs: torch.tensor, lsd: torch.tensor, ml: float,
                 mc: float):
        self.isRemoved = [False, False, False, False, False, False]
        for i in range(6):
            if lsd[i] > ml:
                self.isRemoved[i] = True
        for i in range(2):
            if self.isRemoved[i] is True:
                continue
            for j in range(i + 1, 3):
                if self.isRemoved[j] is True:
                    continue
                if self.cosDihedralAngle(outputs[i][0:3],
                                         outputs[j][0:3]) > mc:
                    if lsd[i] > lsd[j]:
                        self.isRemoved[i] = True
                    else:
                        self.isRemoved[j] = True

        for i in range(3, 5, 1):
            if self.isRemoved[i] is True:
                continue
            for j in range(i + 1, 6):
                if self.isRemoved[j] is True:
                    continue
                if self.cosDihedralAngle(outputs[i][1:4],
                                         outputs[j][1:4]) > mc:
                    if lsd[i] > lsd[j]:
                        self.isRemoved[i] = True
                    else:
                        self.isRemoved[j] = True

        for i in range(6):
            if self.isRemoved[i] is True:
                outputs[i] = torch.zeros(4)
        return outputs

    def cosDihedralAngle(self, normal1: torch.tensor, normal2: torch.tensor):
        return torch.abs(
            torch.dot(normal1, normal2) /
            (torch.norm(normal1) * torch.norm(normal2)))
