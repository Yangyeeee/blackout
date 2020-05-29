from __future__ import print_function

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse
import csv
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from time import localtime
import time
from torch.autograd import Variable
from models.LeNet5_lgm import *
from models.resnet import *
from models.PreActResNet import *
from COT import *
from GCE import *
from utils import *
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser(
    description='PyTorch training using GuidedComplementEntropy')
parser.add_argument('--GCE', action='store_true',
                    help='Using GuidedComplementEntropy')
parser.add_argument('--black', action='store_true',
                    help='Using GuidedComplementEntropy')
parser.add_argument('--cifar', default=0, type=int,
                    help='Using GuidedComplementEntropy')
parser.add_argument('--alpha', '-a', default=0.333, type=float,
                    help='alpha for guiding factor')
parser.add_argument('--k',  default=5, type=int,
                    help='alpha for guiding factor')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--seed', default=11111, type=int, help='rng seed')
parser.add_argument('--decay', default=1e-4, type=float,
                    help='weight decay (default=1e-4)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--batch-size', '-b', default=128,
                    type=int, help='mini-batch size (default: 128)')
parser.add_argument('--epochs','-e', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--gpu', type=str, default="0",
                    help='Which gpu to use.')

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
batch_size = args.batch_size
base_learning_rate = args.lr

current_time = time.strftime('%d_%H:%M:%S', localtime())
writer1 = SummaryWriter(log_dir='runs/' + current_time + '_' + args.sess, flush_secs=30)
if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu

if args.cifar == 0:
    # Data (Default: MNIST)
    print('==> Preparing MNIST data.. (Default)')

    # scale to [0, 1] without standard normalize
    transform_train = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(
        root='../../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='../../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=False, num_workers=2)
elif args.cifar == 10:
    print('==> Preparing CIFAR10 data.. (Default)')
    # classes = 10

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='../../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='../../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

elif args.cifar == 100:
    print('==> Preparing CIFAR100 data.. (Default)')
    # classes = 10

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='../../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='../../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7.' +
                            args.sess + '_' + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print('==> Building model.. (Default : LeNet5)')
    start_epoch = 0
    if args.cifar == 0:
        net = LeNet5_MNIST()

    elif args.cifar == 10:
        net = PreActResNet18()  # ##ResNet56(1)# ResNet34()ResNet101()PreActResNet18_CIFAR100()
    elif args.cifar == 100:
        net = PreActResNet18_CIFAR100()

result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + \
    '_' + args.sess + '_' + str(args.seed) + '.csv'

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

if args.cifar == 10 or args.cifar == 100:
    if args.GCE:
        cotcriterion = ComplementEntropy(args.cifar)
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        cotoptimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)
        lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(cotoptimizer, milestones=[100, 150])
    if args.black:
        criterion =  black2(args.k,args.cifar) #nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=base_learning_rate,
                      momentum=0.9, weight_decay=args.decay)
else:
    if args.GCE:

        criterion = ComplementEntropy(args.cifar)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
        # optimizer = optim.SGD(net.parameters(), lr=0.0004, momentum=0.9, weight_decay=0.0005)
    else:
        criterion = black1(args.k) #nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))

# Training

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    #   plt.xlim(xmin=-5,xmax=5)
    #   plt.ylim(ymin=-5,ymax=5)
    plt.text(-4.8, 4.6, "epoch=%d" % epoch)
    plt.savefig('./images/GCE_epoch=%d.jpg' % epoch)
    # plt.draw()
    # plt.pause(0.001)
    plt.close()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    data = []
    label = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # Baseline Implementation
        inputs, targets = Variable(inputs), Variable(targets)
        _,outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()
        # data.append(feat)
        # label.append(targets)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # COT Implementation
        if args.GCE:
            inputs, targets = Variable(inputs), Variable(targets)
            feat, outputs = net(inputs)

            loss = cotcriterion(outputs, targets)
            cotoptimizer.zero_grad()
            loss.backward()
            cotoptimizer.step()

    # datas = torch.cat(data, 0)
    # labels = torch.cat(label, 0)
    # visualize(datas.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
    return (train_loss / batch_idx, 100. * correct / total)


def test(epoch):
    global best_acc
    # net = net.module
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ip1_loader = []
    idx_loader = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            _,outputs = net(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()
            #
            # ip1_loader.append(feat)
            # idx_loader.append((targets))

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # feats = torch.cat(ip1_loader, 0)
    # labels = torch.cat(idx_loader, 0)
    # tsne = TSNE(n_components=2, learning_rate=200).fit_transform(feats.data.cpu())
    # visualize(tsne, labels.data.cpu().numpy(), 0)
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss / batch_idx, 100. * correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7.' +
               args.sess + '_' + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def complement_adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(start_epoch, args.epochs):

    adjust_learning_rate(optimizer, epoch)
    if args.GCE:
        complement_adjust_learning_rate(cotoptimizer, epoch)
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    # lr_scheduler.step()
    # if args.GCE:
    #     lr_scheduler2.step()
    print(best_acc)
    writer1.add_scalar('Loss/test', test_loss, epoch)
    writer1.add_scalar('ACC/test', test_acc, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])


# test_loss, test_acc = test(0)
# print(test_acc)
