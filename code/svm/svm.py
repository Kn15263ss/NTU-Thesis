import os
import time
import argparse
import numpy as np
import pickle
import utils
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.suggest import BayesOptSearch

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument('--Dataset_name', type=str, default='')

args, unparsed = parser.parse_known_args()


class HyperTrain(Trainable):

    def _get_dataset(self, name):

        data_transforms = transforms.Compose([
            transforms.ToTensor()])

        if name == 'FashionMNIST':

            dataset = torchvision.datasets.FashionMNIST(
                root="/home/willy-huang/workspace/data/FashionMNIST",
                transform=data_transforms, download=True)

            return dataset

        elif name == 'KMNIST':

            dataset = torchvision.datasets.KMNIST(
                root="/home/willy-huang/workspace/data/KMNIST",
                transform=data_transforms, download=True)

            return dataset

        elif name == 'CIFAR10':

            dataset = torchvision.datasets.CIFAR10(
                root="/home/willy-huang/workspace/data/CIFAR10/",
                transform=data_transforms, download=True)

            return dataset

        elif name == 'SVHN':

            dataset = torchvision.datasets.SVHN(
                root="/home/willy-huang/workspace/data/SVHN/",
                transform=data_transforms, download=True)

            return dataset

        elif name == 'STL10':

            dataset = torchvision.datasets.STL10(
                root="/home/willy-huang/workspace/data/STL10/",
                transform=data_transforms, download=True)

            return dataset

    def _setup(self, config):

        self.name = args.Dataset_name
        data = self._get_dataset(self.name)

        self.c = config["c"]
        self.gamma = config["gamma"]

        train_loader = DataLoader(data, batch_size=len(data))
        train = next(iter(train_loader))[0].numpy()
        label = next(iter(train_loader))[1].numpy()

        train = train.reshape(train.shape[0], -1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            train, label, test_size=0.2, random_state=42)

        self.seed_table = np.array(
            ["", "c", "gamma", "accuracy"])

        loggername = '{}.log'.format(self.name)
        self.logger = utils.buildLogger(loggername)

        self.logger.info("Date setup Done")

    def _train(self):

        self.start_time = time.time()
        self.model = SVC(kernel='rbf', C=self.c, gamma=self.gamma)
        self.model.fit(self.X_train, self.y_train)
        return self._test()

    def _test(self):

        accuracy = self.model.score(self.X_test, self.y_test)
        self.logger.info("accuracy: {}, time = {}".format(
            accuracy, (time.time() - self.start_time)))

        save = np.array(
            [self.seed_table,
             [str(self.name),
              str(self.c),
              str(self.gamma),
              str(accuracy)]])

        np.savetxt("./seed(50).csv", save, delimiter=',', fmt="%s")

        return {"mean_loss": 1-accuracy, "mean_accuracy": accuracy}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "final_model.pickle")
        file = open(checkpoint_path, "wb")
        pickle.dump(self.model, file)
        file.close()
        return checkpoint_path

    def _restore(self, checkpoint_path):
        file = open(checkpoint_path, "rb")
        self.model = pickle.load(file)
        file.close()


if __name__ == "__main__":

    ray.init(num_cpus=4)

    space = {
        "c": (-3, 5),
        "gamma": (-3, 4),
    }

    algo = BayesOptSearch(
        space,
        max_concurrent=4,
        reward_attr="neg_mean_loss",
        utility_kwargs={
            "kind": "ei",
            "kappa": 2.5,
            "xi": 0.01
        },
        random_state="special_2"
    )

    exp_name = "{}.svm_result".format(args.Dataset_name)
    tune.run_experiments(
        {
            exp_name: {
                "stop": {
                    # "mean_accuracy": 0.98,
                    "training_iteration": 1
                    # "epoch": 50
                },
                "resources_per_trial": {
                    "cpu": 1
                },
                "run": HyperTrain,
                "checkpoint_at_end": True,
                "num_samples": 1,
                "config": {
                    "c": tune.grid_search([0.01, 0.1, 1, 10, 100]),
                    "gamma": tune.grid_search([0.01, 0.1, 1, 10, 100]),
                    # "c": tune.sample_from(
                    #     lambda spec: 10 ** -np.random.uniform(-3, 5)),
                    # "gamma": tune.sample_from(
                    #     lambda spec: 10 ** -np.random.uniform(-3, 4)),
                },
                "local_dir": "/home/willy-huang/workspace/research/ray_results",
            }
        },
        resume=True,
        verbose=1,
        # search_alg=algo
    )
