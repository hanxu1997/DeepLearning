# 用于cifar数据集的AlexNet网络
# 简化后全连接层仅一层 256->10
import argparse
import os
import shutil
import time
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
#把Tensor转成Image，方便可视化
show=ToPILImage()
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig


from numpy import*
# 上下翻转
def TurnUpDown(a):
    return a[::-1]
# 左右翻转
def TurnLeftRight(a):
    return numpy.array(list(map(TurnUpDown, a)))
# 翻转180度
def Turn180(a):
    return numpy.array(TurnUpDown(list(map(TurnUpDown, a))))
# 三维矩阵下后两维左右翻转
def Turn3D(crop):
    crop = crop.transpose(0,2,1)
    crop = TurnLeftRight(crop)
    crop = crop.transpose(0,2,1)
    return crop
# 4D 左右翻转
def Turn4D(crop):
    return crop

 # batchSize 未实现!
 # inputs: 128*256*3*3
 # batchSize: 128
 # channel: 256
 # inputSize: 5
 # cropSize: 3
 # crops: 10*128*256*3*3
def tencrop(inputs, batchSize, channel, inputSize, cropSize):
    crops = arange(10*batchSize*channel*cropSize*cropSize)
    crops = crops.reshape(10, batchSize, channel, cropSize, cropSize)
    edgestart = inputSize-cropSize;
    midstart = int(floor(inputSize/2)-floor(cropSize/2))
    midend = midstart+cropSize;

    crops[0] = inputs[:,:,0:cropSize,0:cropSize]
    crops[1] = inputs[:,:,0:cropSize,edgestart:inputSize]
    crops[2] = inputs[:,:,edgestart:inputSize,0:cropSize]
    crops[3] = inputs[:,:,edgestart:inputSize,edgestart:inputSize]
    crops[4] = inputs[:,:,midstart:midend,midstart:midend]
    crops[5] = Turn4D(crops[0])
    crops[6] = Turn4D(crops[1])
    crops[7] = Turn4D(crops[2])
    crops[8] = Turn4D(crops[3])
    crops[9] = Turn4D(crops[4])
    # return crops
    return crops

 # no batchSize 测试通过
 # inputs: 256*3*3
 # channel: 256
 # inputSize: 5
 # cropSize: 3
 # crops: 10*256*3*3
def tencrop1(inputs, channel, inputSize, cropSize):
    crops = arange(10*channel*cropSize*cropSize)
    crops = crops.reshape(10, channel, cropSize, cropSize)
    edgestart = inputSize-cropSize;
    midstart = int(floor(inputSize/2)-floor(cropSize/2))
    midend = midstart+cropSize;

    crops[0] = inputs[:,0:cropSize,0:cropSize]
    crops[1] = inputs[:,0:cropSize,edgestart:inputSize]
    crops[2] = inputs[:,edgestart:inputSize,0:cropSize]
    crops[3] = inputs[:,edgestart:inputSize,edgestart:inputSize]
    crops[4] = inputs[:,midstart:midend,midstart:midend]
    crops[5] = Turn3D(crops[0])
    crops[6] = Turn3D(crops[1])
    crops[7] = Turn3D(crops[2])
    crops[8] = Turn3D(crops[3])
    crops[9] = Turn3D(crops[4])
    return crops


a = arange(25)
a = a.reshape(5,5)
# five crop
a1 = a[0:3,0:3]
a2 = a[0:3,2:5]
a3 = a[2:5,0:3]
a4 = a[2:5,2:5]
a5 = a[1:4,1:4]
b = zeros((256,5,5))
for i in range(256):
    b[i] = a

crops = tencrop1(b, 256, 5, 3)
for i in range(10):
    print(crops[i])
    print(shape(crops[i]))





# cropSize = 3
# featureMapNumpy = featureMapBatch.data.numpy()
# batchSize = numpy.size(featureMapNumpy,0)
# channel = numpy.size(featureMapNumpy,1)
# featureSize = numpy.size(featureMapNumpy,2)

# print(type(train_outputs))
# Variable(torch.from_numpy(tenFeatureMap))
