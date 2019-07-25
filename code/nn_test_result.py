import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from model import Vgg11, Resnet18, MobileNet_V2


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
            train=False,
            transform=data_transforms)
        num_classes = 10
        input_size = 512 * 1 * 1

        return dataset, num_classes, input_size

    if name == 'KMNIST':

        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize])
        dataset = torchvision.datasets.KMNIST(
            root="/home/willy-huang/workspace/data/KMNIST",
            train=False,
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
            train=False,
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
            split="test",
            transform=data_transforms,
            download=True)
        num_classes = 10
        input_size = 512 * 1 * 1

        return dataset, num_classes, input_size

    elif name == 'STL10':

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        dataset = torchvision.datasets.STL10(
            root="/home/willy-huang/workspace/data/STL10/",
            split="test",
            transform=data_transforms)
        num_classes = 10
        input_size = 512 * 3 * 3

        return dataset, num_classes, input_size


def evaluate(name, nnArchitecture, model_path):

    dataset, num_class, input_size = get_dataset(name)
    test_ld = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=False)

    if nnArchitecture == 'Vgg11':
        model = Vgg11(num_class, input_size).cuda()

    elif nnArchitecture == 'Resnet18':
        model = Resnet18(num_class, input_size).cuda()

    elif nnArchitecture == 'MobileNet_V2':
        model = MobileNet_V2(num_class, input_size).cuda()

    else:
        model = None
        assert 0

    modelCheckpoint = torch.load(model_path)
    state_dict = modelCheckpoint['state_dict']

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    print("Build Model Done")

    model.eval()

    loss = nn.CrossEntropyLoss()

    lossVal = 0
    lossValNorm = 0
    correct = 0

    num_samples = 0
    for batchID, (input, target) in enumerate(test_ld):
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

    outLoss = lossVal / lossValNorm
    accuracy = correct.item() / num_samples

    print("accuracy: {:.5f}".format(accuracy))
    print('Proccess Done!')
    return accuracy


def main():

    base_path = "/home/willy-huang/workspace/research/test/best_parameter"
    files = os.listdir(base_path)

    for f in sorted(files):
        name = f
        path = os.path.join(base_path, f)
        for p in os.listdir(path):
            nnArchitecture = p
            path2 = os.path.join(path, p)
            for p2 in os.listdir(path2):
                path3 = os.path.join(path2, p2, 'valid')
                for p3 in os.listdir(path3):
                    if p3.endswith(".pth.tar"):
                        model_path = os.path.join(path3, p3)
                        print("Dataset: {}, nn_Net: {}, Method: {}".format(
                              name, nnArchitecture, p2))
                        accuracy = evaluate(name, nnArchitecture, model_path)
                        print('')
                        test_path = os.path.join(path2, p2, 'test')

                        if not os.path.exists(test_path):
                            os.mkdir(test_path)
                        with open(test_path+'/'+'test_result.txt', 'w') as f:
                            f.write(str(accuracy))


if __name__ == "__main__":
    main()
