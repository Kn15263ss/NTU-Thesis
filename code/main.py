import os
import random
import numpy as np
import time
import argparse
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import ray
from hyperopt import hp
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.suggest import BayesOptSearch
from model import Vgg11, Resnet18, MobileNet, MobileNet_V2
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser()
parser.add_argument('--Dataset_name', type=str, default='FashionMNIST')
parser.add_argument('--Network_name', type=str, default='Vgg11')

args, unparsed = parser.parse_known_args()


class HyperTrain(Trainable):

    def _get_dataset(self, name):

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
                root="/home/kn15263s/data/FashionMNIST",
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
                root="/home/kn15263s/data/KMNIST",
                transform=data_transforms, download=True)
            num_classes = 10
            input_size = 512 * 1 * 1

            return dataset, num_classes, input_size

        elif name == 'CIFAR10':

            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            dataset = torchvision.datasets.CIFAR10(
                root="/home/kn15263s/data/CIFAR10/",
                transform=data_transforms)
            num_classes = 10
            input_size = 512 * 1 * 1

            return dataset, num_classes, input_size

        elif name == 'SVHN':

            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            dataset = torchvision.datasets.SVHN(
                root="/home/kn15263s/data/SVHN/",
                transform=data_transforms)
            num_classes = 10
            input_size = 512 * 1 * 1

            return dataset, num_classes, input_size

        elif name == 'STL10':

            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            dataset = torchvision.datasets.STL10(
                root="/home/kn15263s/data/STL10/",
                transform=data_transforms)
            num_classes = 10
            input_size = 512 * 3 * 3

            return dataset, num_classes, input_size

        # elif name == 'Food':
        #
        #     class Food(Dataset):
        #
        #         def __init__(self, files, class_names, transform=transforms.ToTensor()):
        #
        #             self.data = files
        #             self.transform = transform
        #             self.class_names = class_names
        #
        #         def __getitem__(self, idx):
        #             img = Image.open(self.data[idx]).convert('RGB')
        #             name = self.data[idx].split('/')[-2]
        #             y = self.class_names.index(name)
        #             img = self.transform(img)
        #             return img, y
        #
        #         def __len__(self):
        #             return len(self.data)
        #
        #     data_transforms = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         normalize])
        #
        #     path = '/home/willy-huang/workspace/data/food'
        #     files_training = glob(os.path.join(path, '*/*.jpg'))
        #     class_names = []
        #
        #     for folder in os.listdir(os.path.join(path)):
        #         class_names.append(folder)
        #
        #     num_classes = len(class_names)
        #     dataset = Food(files_training, class_names, data_transforms)
        #     input_size = 512 * 7 * 7
        #
        #     return dataset, num_classes, input_size
        #
        # elif name == 'Stanford_dogs':
        #
        #     class Stanford_dogs(Dataset):
        #
        #         def __init__(self, files, class_names, transform=transforms.ToTensor()):
        #
        #             self.data = files
        #             self.transform = transform
        #             self.class_names = class_names
        #
        #         def __getitem__(self, idx):
        #             img = Image.open(self.data[idx]).convert('RGB')
        #             name = self.data[idx].split('/')[-2]
        #             y = self.class_names.index(name)
        #             img = self.transform(img)
        #             return img, y
        #
        #         def __len__(self):
        #             return len(self.data)
        #
        #
        #     data_transforms = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         normalize])
        #
        #     path = '/home/willy-huang/workspace/data/stanford_dogs'
        #     files_training = glob(os.path.join(path, '*/*.jpg'))
        #     class_names = []
        #
        #     for folder in os.listdir(os.path.join(path)):
        #         class_names.append(folder)
        #
        #     num_classes = len(class_names)
        #     dataset = Stanford_dogs(files_training, class_names, data_transforms)
        #     input_size = 512 * 7 * 7
        #
        #     return dataset, num_classes, input_size

    def _setup(self, config):
        random.seed(50)
        np.random.seed(50)
        torch.cuda.manual_seed_all(50)
        torch.manual_seed(50)
        self.total_time = time.time()
        self.name = args.Dataset_name
        nnArchitecture = args.Network_name

        dataset, num_class, input_size = self._get_dataset(self.name)

        num_total = len(dataset)
        shuffle = np.random.permutation(num_total)
        split_val = int(num_total * 0.2)

        train_idx, valid_idx = shuffle[split_val:], shuffle[:split_val]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.trainset_ld = DataLoader(
            dataset, batch_size=256, sampler=train_sampler, num_workers=4)
        self.validset_ld = DataLoader(
            dataset, batch_size=256, sampler=valid_sampler, num_workers=4)

        self.modelname = '{}--{}.pth.tar'.format(self.name, nnArchitecture)
        loggername = self.modelname.replace("pth.tar", "log")
        self.logger = utils.buildLogger(loggername)

        self.seed_table = np.array(
            ["", "epoch", "lr", "momentum", "weight_decay", "factor",
             "outLoss", "accuracy"])

        # ---- hyperparameters ----
        self.lr = config["lr"]
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]
        self.factor = config["factor"]

        self.epochID = 0
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = -999999999999.0

        # -------------------- SETTINGS: NETWORK ARCHITECTURE

        if nnArchitecture == 'Vgg11':
            self.model = Vgg11(num_class, input_size).cuda()

        elif nnArchitecture == 'Resnet18':
            self.model = Resnet18(num_class, input_size).cuda()

        elif nnArchitecture == 'MobileNet':
            self.model = MobileNet(num_class, input_size).cuda()

        elif nnArchitecture == 'MobileNet_V2':
            self.model = MobileNet_V2(num_class, input_size).cuda()

        else:
            self.model = None
            assert 0

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.logger.info("Build Model Done")

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER --------------------
        self.optimizer = optim.SGD(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
            nesterov=False)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=self.factor, patience=10, mode='min')

        self.logger.info("Build Optimizer Done")

    def _train_iteration(self):
        self.start_time = time.time()
        self.model.train()

        losstra = 0
        losstraNorm = 0

        for batchID, (input, target) in enumerate(self.trainset_ld):
            varInput = Variable(input).cuda()
            varTarget = Variable(target).cuda()
            varOutput = self.model(varInput)

            lossvalue = self.loss(varOutput, varTarget)

            losstra += lossvalue.item()
            losstraNorm += 1

            self.optimizer.zero_grad()
            lossvalue.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
            self.optimizer.step()

        self.trainLoss = losstra / losstraNorm

    def _test(self):

        self.model.eval()

        lossVal = 0
        lossValNorm = 0
        correct = 0

        num_samples = 0
        for batchID, (input, target) in enumerate(self.validset_ld):
            with torch.no_grad():
                varInput = Variable(input).cuda(async=True)
                varTarget = Variable(target).cuda(async=True)
                varOutput = self.model(varInput)

                losstensor = self.loss(varOutput, varTarget)

                pred = varOutput.argmax(1)
                correct += (pred == varTarget).sum().cpu()

                lossVal += losstensor.item()
                lossValNorm += 1
                num_samples += len(input)

        self.outLoss = lossVal / lossValNorm
        accuracy = correct.item() / num_samples

        self.scheduler.step(self.outLoss, epoch=self.epochID)

        if accuracy > self.accuracy:
            self.accuracy = accuracy

            torch.save({'epoch': self.epochID + 1,
                        'state_dict': self.model.state_dict(),
                        'loss': self.outLoss,
                        'best_accuracy': self.accuracy,
                        'optimizer': self.optimizer.state_dict(),
                        }, "./best_"+self.modelname)

            save = np.array(
                [self.seed_table,
                 [str(self.name),
                  str(self.epochID + 1),
                  str(self.lr),
                  str(self.momentum),
                  str(self.weight_decay),
                  str(self.factor),
                  str(self.outLoss),
                  str(self.accuracy)]])

            np.savetxt("./seed(50).csv", save, delimiter=',', fmt="%s")

        self.logger.info('Epoch [' + str(self.epochID + 1) + '] loss= {:.5f}'.format(self.outLoss) +
                         ' ---- accuracy= {:.5f}'.format(accuracy) +
                         ' ---- best_accuracy= {:.5f}'.format(self.accuracy) +
                         ' ---- model: {}'.format(self.modelname) +
                         ' ---- time: {:.1f} s'.format((time.time() - self.start_time)) +
                         ' ---- total_time: {:.1f} s'.format((time.time() - self.total_time)))

        self.epochID += 1
        return {"episode_reward_mean": accuracy, "neg_mean_loss":self.outLoss, "mean_accuracy": accuracy, "epoch": self.epochID, 'mean_train_loss': self.trainLoss}

    def _train(self):
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save({
            "epoch": self.epochID,
            "best_accuracy": self.accuracy,
            'loss': self.outLoss,
            "state_dict": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


if __name__ == "__main__":

    ray.init(num_cpus=4, num_gpus=4)

    sched = HyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        max_t=50)

    space = {
        "factor": (0.01, 0.99),
        "lr": (1, 5),
        "momentum": (0, 0.99),
        "weight_decay": (1, 5)
    }

    algo = BayesOptSearch(
        space,
        max_concurrent=1,
        reward_attr="neg_mean_loss",
        utility_kwargs={
            "kind": "ei",
            "kappa": 2.5,
            "xi": 0.01
        },
        random_state="special_1"
    )

    exp_name = "{}.{}_result".format(args.Dataset_name, args.Network_name)
    tune.run_experiments(
        {
            exp_name: {
                "stop": {
                    # "mean_accuracy": 0.98,
                    # "training_iteration": 99999
                    "epoch": 5
                },
                "resources_per_trial": {
                    "cpu": 1,
                    "gpu": 1
                },
                "run": HyperTrain,
                "checkpoint_at_end": True,
                "num_samples": 32,
                "config": {
                    # "lr": tune.grid_search([0.01, 0.1]),
                    # "momentum": tune.grid_search([0.1, 0.9]),
                    # "weight_decay": tune.grid_search([1e-4, 1e-6]),
                    # "factor": tune.grid_search([0.1, 0.5])
                    # "factor": tune.sample_from(
                    #      lambda spec: np.random.uniform(0.01, 0.99)),
                    # "lr": tune.sample_from(
                    #      lambda spec: 10 ** -np.random.uniform(low=1, high=5)),
                    # "momentum": tune.sample_from(
                    #      lambda spec: np.random.uniform(0, 0.99)),
                    # "weight_decay": tune.sample_from(
                    #      lambda spec: np.random.choice(np.concatenate((10 ** -np.random.uniform(low=1, high=5, size=100), [0])))),
                    # "lr": tune.grid_search([1.9093900968826365e-05, 0.01134481828504152, 0.004356458230295825, 0.0010739647148905634]),
                    # "momentum": tune.grid_search([0.31422975883190635, 0.42373653911421866]),
                    # "weight_decay": tune.grid_search([0.002221483083758635, 5.3161487511679675e-05]),
                    # "factor": tune.grid_search([0.5365368078702403, 0.03781501025511687])
                },
                "local_dir": "/home/kn15263s/workspace/Bayesian_32_results",
            }
        },
        resume=True,
        verbose=1,
        search_alg=algo,
        # scheduler=sched
    )
