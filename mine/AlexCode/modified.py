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


#########################载入数据集

# load and normalizing the cifar10 training and test datasets 
# normalize to [-1,1] Tensor**
# combine module ToTensor and Normalize
print('==> Preparing dataset')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

####数据预处理
# train
transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4), # 随机剪裁
                                transforms.RandomHorizontalFlip(), # 随机水平翻转
                                transforms.ToTensor(),#转为tensor
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#归一化
                            ])
# test
transform_test = transforms.Compose([
                                transforms.RandomHorizontalFlip(), # 随机水平翻转
                                transforms.ToTensor(),#转为tensor
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#归一化
                            ])
####加载数据
#训练集
trainset=datasets.CIFAR10(root='./data',
                            train=True,
                            download=True,
                            transform=transform)

trainloader=data.DataLoader(trainset,
                            batch_size=128,
                            shuffle=True,
                            num_workers=0)
#测试集
testset=datasets.CIFAR10(root='./data',
                            train=False,
                            download=True,
                            transform=transform_test)

testloader=data.DataLoader(testset,
                            batch_size=100,
                            shuffle=True,
                            num_workers=0)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# print the class of the first image
(data,label)=trainset[0]
print("firstclass: "+classes[label])
print(type(data))
print(data.size())
 
#########################定义网络
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
  

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        # self.classifier = nn.Linear(256*5*5, num_classes)

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x



import torch.optim as optim

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx == 0:
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            print('inputs: ', type(inputs))
            # compute output
            outputs = model(inputs)
            print('outputs: ', type(outputs))
            print(outputs.size())
    return (outputs)



def adjust_learning_rate(optimizer, epoch):
    if epoch in [81, 122]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


best_acc = 0  # best test accuracy
# Model
print("==> creating model alexnet")
model = AlexNet();
num_classes = 10;
use_cuda = False
# use_cuda = True
if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = model.cuda()

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


# Train and val
max_epoch = 1

for epoch in range(0, max_epoch):
    adjust_learning_rate(optimizer, epoch)
    for param_group in optimizer.param_groups:
        lr_show = param_group['lr']
    print('\nEpoch: [%d | %d] %f' % (epoch + 1, max_epoch, lr_show))
    train_outputs = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
    train_outputs = train_outputs.data.numpy()
    print(type(train_outputs))
    print(numpy.shape(train_outputs))
    print(numpy.size(train_outputs,0))
    print(numpy.size(train_outputs,1))
    print(numpy.size(train_outputs,2))
    print(numpy.size(train_outputs,3))

 # Variable:featureMap
def tenCrops(featureMapBatch):
    cropSize = 3
    featureMapNumpy = featureMapBatch.data.numpy()
    batchSize = numpy.size(featureMapNumpy,0)
    channel = numpy.size(featureMapNumpy,1)
    featureSize = numpy.size(featureMapNumpy,2)
    
    print(type(train_outputs))
    # Variable(torch.from_numpy(tenFeatureMap))
    return tenFeatureMap