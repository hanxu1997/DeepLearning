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
transform = transforms.Compose([
                                # transforms.RandomCrop(32, padding=4), # 随机剪裁
                                # transforms.RandomHorizontalFlip(), # 随机水平翻转
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
                            transform=transform)

testloader=data.DataLoader(testset,
                            batch_size=100,
                            shuffle=True,
                            num_workers=0)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# print the class of the first image
(data,label)=trainset[0]
print("firstclass: "+classes[label])

 
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



import torch.optim as optim

trainRecord = open("trainRecord_LR_1.csv", "w")

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
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch=batch_idx + 1
        size=len(trainloader)
        data=data_time.avg
        bt=batch_time.avg
        loss=losses.avg
        top1value=top1.avg
        top5value=top5.avg


        if (batch == size):
            print('train[%03d/%d] Data: %.3fs | Batch: %.3fs | Loss: %.4f | top1: %.4f | top5: %.4f' % (
                        batch,
                        size,
                        data,
                        bt,
                        loss,
                        top1value,
                        top5value
                        ))
            print('%.4f,%.4f,%.4f' % (
            loss,
            top1value,
            top5value
            ), file = trainRecord)
    return (losses.avg, top1.avg)

testRecord = open("testRecord_LR_1.csv", "w")

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        batch=batch_idx + 1
        size=len(testloader)
        data=data_time.avg
        bt=batch_time.avg
        loss=losses.avg
        top1value=top1.avg
        top5value=top5.avg
        if (batch == size):  
            print('test [%03d/%d] Data: %.3fs | Batch: %.3fs | Loss: %.4f | top1: %.4f | top5: %.4f' % (
                        batch,
                        size,
                        data,
                        bt,
                        loss,
                        top1value,
                        top5value
                        ))
            print('%.4f,%.4f,%.4f' % (
            loss,
            top1value,
            top5value
            ), file = testRecord)
    return (losses.avg, top1.avg)


def adjust_learning_rate(optimizer, epoch):
    if epoch in [81, 122]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


best_acc = 0  # best test accuracy
# Model
print("==> creating model alexnet")
model = AlexNet();
num_classes = 10;
# use_cuda = False
use_cuda = True
if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = model.cuda()

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


# Train and val
max_epoch = 164

for epoch in range(0, max_epoch):
    adjust_learning_rate(optimizer, epoch)
    for param_group in optimizer.param_groups:
        lr_show = param_group['lr']
    print('\nEpoch: [%d | %d] %f' % (epoch + 1, max_epoch, lr_show))
    train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
    test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
    best_acc = max(test_acc, best_acc)

print('Best acc:')
print(best_acc)
trainRecord.close()
testRecord.close()
