import os
import numpy as np
import time
import argparse
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import Vgg11, Resnet18, MobileNet_V2
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from tensorboardX import SummaryWriter

torch.cuda.manual_seed_all(50)


def get_dataset(name):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    if name == 'FashionMNIST':

        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize])
        dataset = torchvision.datasets.FashionMNIST(
            root="/home/willy-huang/workspace/data/FashionMNIST",
            transform=data_transforms)
        num_classes = 10
        input_size = 512 * 1 * 1

        return dataset, num_classes, input_size

    elif name == 'KMNIST':

        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize])

        dataset = torchvision.datasets.KMNIST(
            root="/home/willy-huang/workspace/data/KMNIST",
            transform=data_transforms)
        num_classes = 10
        input_size = 512 * 1 * 1

        return dataset, num_classes, input_size

    elif name == 'CIFAR10':

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        dataset = torchvision.datasets.CIFAR10(
            root="/home/willy-huang/workspace/data/CIFAR10/",
            transform=data_transforms)
        num_classes = 10
        input_size = 512 * 1 * 1

        return dataset, num_classes, input_size

    elif name == 'SVHN':

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        dataset = torchvision.datasets.SVHN(
            root="/home/willy-huang/workspace/data/SVHN/",
            transform=data_transforms)
        num_classes = 10
        input_size = 512 * 1 * 1

        return dataset, num_classes, input_size

    elif name == 'STL10':

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        dataset = torchvision.datasets.STL10(
            root="/home/willy-huang/workspace/data/STL10/",
            transform=data_transforms)
        num_classes = 10
        input_size = 512 * 3 * 3

        return dataset, num_classes, input_size


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset_name', type=str, default='')
    parser.add_argument('--Network_name', type=str, default='')
    parser.add_argument('--lr', type=float, default='')
    parser.add_argument('--momentum', type=float, default='')
    parser.add_argument('--factor', type=float, default='')
    parser.add_argument('--weight_decay', type=float, default='')

    args, unparsed = parser.parse_known_args()

    name = args.Dataset_name
    nnArchitecture = args.Network_name

    dataset, num_class, input_size = get_dataset(name)

    num_total = len(dataset)
    shuffle = np.random.permutation(num_total)
    split_val = int(num_total * 0.2)

    train_idx, valid_idx = shuffle[split_val:], shuffle[:split_val]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainset_ld = DataLoader(
        dataset, batch_size=256, sampler=train_sampler, num_workers=4)
    validset_ld = DataLoader(
        dataset, batch_size=256, sampler=valid_sampler, num_workers=4)

    modelname = '{}--{}--{:.6f}.pth.tar'.format(
        name, nnArchitecture, args.weight_decay)

    dirpath = os.path.join(
        "./test_nn/", modelname.replace(".pth.tar", ""))

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    loggername = os.path.join(
        dirpath, modelname.replace("pth.tar", "log"))
    logger = utils.buildLogger(loggername)

    writer = SummaryWriter(dirpath)

    seed_table = np.array(
        [["train_accuracy", "train_loss", "valid_accuracy", "valid_loss"]])

    # ---- hyperparameters ----
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    factor = args.factor

    epoch = 50
    loss = nn.CrossEntropyLoss()

    # -------------------- SETTINGS: NETWORK ARCHITECTURE

    if nnArchitecture == 'Vgg11':
        model = Vgg11(num_class, input_size).cuda()

    elif nnArchitecture == 'Resnet18':
        model = Resnet18(num_class, input_size).cuda()

    elif nnArchitecture == 'MobileNet_V2':
        model = MobileNet_V2(num_class, input_size).cuda()

    else:
        model = None
        assert 0

    model = torch.nn.DataParallel(model).cuda()
    logger.info("Build Model Done")

    # -------------------- SETTINGS: OPTIMIZER & SCHEDULER --------------------
    optimizer = optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=lr, momentum=momentum, weight_decay=weight_decay,
        nesterov=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=factor, patience=10, mode='min')

    logger.info("Build Optimizer Done")

    for epochID in range(0, epoch):

        model.train()

        losstra = 0
        losstraNorm = 0
        correct = 0

        num_samples = 0

        for batchID, (input, target) in enumerate(trainset_ld):
            varInput = Variable(input).cuda(async=True)
            varTarget = Variable(target).cuda(async=True)
            varOutput = model(varInput)

            lossvalue = loss(varOutput, varTarget)

            pred = varOutput.argmax(1)
            correct += (pred == varTarget).sum().cpu()

            losstra += lossvalue.item()
            losstraNorm += 1
            num_samples += len(input)

            optimizer.zero_grad()
            lossvalue.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            optimizer.step()

        train_outLoss = losstra / losstraNorm
        train_accuracy = correct.item() / num_samples

        model.eval()

        lossVal = 0
        lossValNorm = 0
        correct = 0

        num_samples = 0
        for batchID, (input, target) in enumerate(validset_ld):
            with torch.no_grad():
                varInput = Variable(input).cuda(async=True)
                varTarget = Variable(target).cuda(async=True)
                varOutput = model(varInput)

                losstensor = loss(varOutput, varTarget)

                pred = varOutput.argmax(1)
                correct += (pred == varTarget).sum().cpu()

                lossVal += losstensor.item()
                lossValNorm += 1
                num_samples += len(input)

        valid_outLoss = lossVal / lossValNorm
        valid_accuracy = correct.item() / num_samples

        writer.add_scalar('train_accuracy', train_accuracy, epochID)
        writer.add_scalar('train_loss', train_outLoss, epochID)
        writer.add_scalar('valid_accuracy', valid_accuracy, epochID)
        writer.add_scalar('valid_loss', valid_outLoss, epochID)

        scheduler.step(valid_outLoss, epoch=epochID)

        logger.info('Epoch [' + str(epochID + 1) + '] loss= {:.5f}'.format(
            valid_outLoss) + ' ---- accuracy= {:.5f}'.format(
            valid_accuracy) + ' ---- model: {}'.format(modelname))

        seed_table = np.append(seed_table,
                               [[str(train_accuracy),
                                 str(train_outLoss),
                                 str(valid_accuracy),
                                 str(valid_outLoss)]], axis=0)

    np.savetxt(
        os.path.join(dirpath, "seed(50).csv"),
        seed_table, delimiter=',', fmt="%s")


if __name__ == "__main__":
    main()
