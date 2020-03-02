import easydict

__C = easydict.EasyDict()
cfg = __C

__C.dataDir = ".\\data"
__C.trainProportion = 0.8
__C.dataSize = 145
__C.negativeSlope = 0.2
__C.modelDir = ".\\model"
__C.resultDir = ".\\result"
