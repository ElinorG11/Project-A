'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import copy

import os
import argparse

from models import resnet
from utils import progress_bar

from util.cutout import Cutout

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0729, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--analog_aware', default=False, action='store_true', help='analog aware training')
parser.add_argument('--alpha', default=2, type=float, help='noise parameter')

# Add Cutoff parameters
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # Using standard normalization. Tried to change normalization as specified here https://github.com/uoguelph-mlrg/Cutout/blob/master/train.py
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Apply Cutout as described arXiv:1708.04552v2
if args.cutout:
    transform_train.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # Using standard normalization. Tried to change normalization as specified here https://github.com/uoguelph-mlrg/Cutout/blob/master/train.py
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)  # Having trouble using batch of 128 reduced to 64

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = resnet.ResNet34()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9)
lambda1 = lambda epoch: 0.9 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
scheduler.last_epoch=500

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    c=0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        parameters, index = [], 0 # START SECTION ADDING NOISE DURING TRAINING

        # Add noise
        if args.analog_aware:
            with torch.no_grad():
                for name, param in net.named_parameters():
                    if (name == 'module.conv1.weight') or (name == 'module.linear.weight'):
                        continue
                    if ('weight' in name) or ('bias' in name):
                        # Set aside ideal weights for backpropagation
                        weight = copy.deepcopy(param.data)
                        parameters.append(weight)

                        noise = torch.normal(mean=0.0, std=0.02 * torch.max(torch.abs(param)) * torch.ones_like(param))
                        param.add_(noise) # END SECTION ADDING NOISE DURING TRAINING

        c = c + 1
        if c == 1:
          print(parameters, file=open('output_store.txt','w'))
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # Restore weights only, without gradients
        if args.analog_aware:  # START SECTION RESTORING IDEAL WEIGHTS
            with torch.no_grad():
                for name, param in net.named_parameters():
                    if (name == 'module.conv1.weight') or (name == 'module.linear.weight'):
                        continue
                    if ('weight' in name) or ('bias' in name):
                        # Set aside ideal weights for backpropagation
                        param.data = parameters[index]
                        index = index + 1  # END SECTION RESTORING IDEAL WEIGHTS
                        
                        if c == 1:
                          print("updating parameters")
                        
        if c == 1:
          print(parameters, file=open('output_load.txt', 'w'))

        loss.backward()
        optimizer.step()

        if args.analog_aware:  # START SECTION WEIGHTS CLIPPING
            with torch.no_grad():
                for name, param in net.named_parameters():
                    if (name == 'module.conv1.weight') or (name == 'module.linear.weight'):
                        continue
                    if ('weight' in name) or ('bias' in name):
                        std = torch.std(param).item()
                        param.clamp(min=-args.alpha * std, max=args.alpha * std)  # END SECTION WEIGHTS CLIPPING

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            parameters, index = [], 0  # START SECTION ADDING NOISE DURING TEST

            # Add noise
            if args.analog_aware:
                with torch.no_grad():
                    for name, param in net.named_parameters():
                        if (name == 'module.conv1.weight') or (name == 'module.linear.weight'):
                            continue
                        if ('weight' in name) or ('bias' in name):
                            # Set aside ideal weights for backpropagation
                            parameters.append(param.data)

                            noise = torch.normal(mean=0.0, std=0.038 * torch.max(torch.abs(param)) * torch.ones_like(param))
                            param.add_(noise)  # END SECTION ADDING NOISE DURING TEST

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # ===================================== START SECTION RESTORING IDEAL WEIGHTS ==================================== #

            # Restore weights only, without gradients
            if args.analog_aware:
                with torch.no_grad():
                    for name, param in net.named_parameters():
                        if (name == 'module.conv1.weight') or (name == 'module.linear.weight'):
                            continue
                        if ('weight' in name) or ('bias' in name):
                            # Set aside ideal weights for backpropagation
                            param.data = parameters[index]
                            index = index + 1

    # ===================================== END SECTION RESTORING IDEAL WEIGHTS ====================================== #

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler.step()
