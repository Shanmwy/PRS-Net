# PRS-Net

an unofficial implementation of PRS-Net, due to a lack of NVIDIA card, the project works on cpu only

# Dependencies
- `python3.5+`
- `pytorch`
- `easydict`


# Data Preparation
## data were already prepared in /data, there's no need to prepare them on yourself.

ShapeNet data were downloaded and stored in `/data`, run `/data/initialize.bat` to initialize the data with binvox. Concerning the condition of my laptop, only 145 samples form `02691156` class are used in training and test.

# Train
1. check `config.py`
2. run `train.py` with desired hyper-parameters to start training:
```bash
$ python3 train.py --rs
```
Note that the default value of hyper-parameters were set in accordance with the paper. The training log would be stored as `/result/trainLog.txt` as default.



# Test
1. run `test.py` to produce final predictions on the validation set after training is done.
```bash
$ python3 test.py
```
Note that the default value of hyper-parameters were set in accordance with the paper. The training log would be stored as `/result/testLog.txt` as default

# TODO
- train the network with larger dataset

│  config.py
│  dataPreparation.py
│  myQuaternion.py
│  PRSNet.py
│  README.md
│  test.py
│  train.py
│  VoxelPointsDataset.py
│
├─data
│  │  binvox.exe
│  │  dataPreparation.bat
│  │  SurfacePointsSamplingPCL.exe
│  │  train.csv
│  │  val.csv
│  │
│  ├─MatureData
│  │
│  ├─ShapeNetCore.v1
│  │
│  └─visulization_test
│
├─model
│      PRS_Net.pkl
│
└─result
    │  testLog.txt
    │  trainLog.txt
    ├─reflectedTest
    └─reflectedTrain