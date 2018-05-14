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
a = arange(25)
a = a.reshape(5,5)
print(a)
a1 = a[0:3,0:3]
print(a1)
a2 = a[0:3,2:5]
print(a2)



# cropSize = 3
# featureMapNumpy = featureMapBatch.data.numpy()
# batchSize = numpy.size(featureMapNumpy,0)
# channel = numpy.size(featureMapNumpy,1)
# featureSize = numpy.size(featureMapNumpy,2)

# print(type(train_outputs))
# Variable(torch.from_numpy(tenFeatureMap))
