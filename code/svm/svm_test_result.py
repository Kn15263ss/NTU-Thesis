import os
import time
import pickle
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataset(name):

    data_transforms = transforms.Compose([
        transforms.ToTensor()])

    if name == 'FashionMNIST':

        dataset = torchvision.datasets.FashionMNIST(
            root="/home/willy-huang/workspace/data/FashionMNIST",
            transform=data_transforms,
            train=False)

        return dataset

    elif name == 'KMNIST':

        dataset = torchvision.datasets.KMNIST(
            root="/home/willy-huang/workspace/data/KMNIST",
            transform=data_transforms,
            train=False)

        return dataset

    elif name == 'CIFAR10':

        dataset = torchvision.datasets.CIFAR10(
            root="/home/willy-huang/workspace/data/CIFAR10/",
            transform=data_transforms,
            train=False)

        return dataset

    elif name == 'SVHN':

        dataset = torchvision.datasets.SVHN(
            root="/home/willy-huang/workspace/data/SVHN/",
            transform=data_transforms,
            split='test')

        return dataset

    elif name == 'STL10':

        dataset = torchvision.datasets.STL10(
            root="/home/willy-huang/workspace/data/STL10/",
            transform=data_transforms,
            split='test')

        return dataset


def evaluate(name, model_path):

    data = get_dataset(name)

    test_loader = DataLoader(data, batch_size=len(data))
    test = next(iter(test_loader))[0].numpy()
    label = next(iter(test_loader))[1].numpy()

    test = test.reshape(test.shape[0], -1)

    file = open(model_path, "rb")
    model = pickle.load(file)
    file.close()
    start_time = time.time()
    accuracy = model.score(test, label)
    print("accuracy: {}, time = {}".format(
        accuracy, (time.time() - start_time)))

    return accuracy


def main():

    base_path = "/home/willy-huang/workspace/research/svm/best_parameter"
    files = os.listdir(base_path)

    for f in sorted(files):
        name = f
        path = os.path.join(base_path, f, 'svm')
        model = 'svm'
        for p in os.listdir(path):
            if p == "randomsearch_12" or p == "randomsearch_24":
                path2 = os.path.join(path, p, 'valid')
                for p2 in os.listdir(path2):
                    if os.path.isdir(os.path.join(path2, p2)):
                        model_path = os.path.join(
                            path2, p2, "final_model.pickle")
                        print("Dataset: {}, Model: {}, Method: {}".format(
                            name, model, p))
                        accuracy = evaluate(name, model_path)
                        print('')
                        test_path = os.path.join(path, p, 'test')

                        if not os.path.exists(test_path):
                            os.mkdir(test_path)
                        with open(os.path.join(test_path, 'test_result.txt'), 'w') as f:
                            f.write(str(accuracy))


if __name__ == "__main__":
    main()
