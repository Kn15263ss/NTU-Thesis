import os
import time
import argparse
import utils
import numpy as np
import pickle
import torchvision
from hyperopt import hp

from torchvision import transforms
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import BayesOptSearch, HyperOptSearch

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
                root="/home/kn15263s/data/FashionMNIST",
                transform=data_transforms)

            return dataset

        elif name == 'CIFAR10':

            dataset = torchvision.datasets.CIFAR10(
                root="/home/kn15263s/data/CIFAR10/",
                transform=data_transforms)

            return dataset

        elif name == 'SVHN':

            dataset = torchvision.datasets.SVHN(
                root="/home/kn15263s/data/SVHN/",
                transform=data_transforms)

            return dataset

        elif name == 'STL10':

            dataset = torchvision.datasets.STL10(
                root="/home/kn15263s/data/STL10/",
                transform=data_transforms)

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

    ray.init(num_cpus=6)

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="neg_mean_loss",
        max_t=100,
        grace_period=5)

    space1 = {
        "c": (-3, 5),
        "gamma": (-3, 4)
    }

    space2 = {
        'factor': hp.uniform('factor', 0.01, 0.999),
        'lr': 10 ** -hp.uniform('lr', 1, 5),
        'momentum': hp.uniform('momentum', 0, 0.99),
        'weight_decay': hp.choice(
            'weight_decay', np.concatenate(
                (10 ** -np.random.uniform(1, 5, size=100),
                 [0])))}

    current_best_params = [
        {
            'factor': 0.1,
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0
        }
    ]

    algo1 = BayesOptSearch(
        space1,
        max_concurrent=6,
        reward_attr="neg_mean_loss",
        utility_kwargs={
            "kind": "ei",
            "kappa": 2.5,
            "xi": 0.01
        },
        random_state="special_2"
    )

    algo2 = HyperOptSearch(
        space2,
        max_concurrent=4,
        reward_attr='neg_mean_loss',
        points_to_evaluate=current_best_params
    )

    exp_name = "{}_result".format(args.Dataset_name)
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
                "num_samples": 48,
                "config": {
#                     "c": tune.grid_search([0.01, 0.1, 1, 10, 100]),
#                     "gamma": tune.grid_search([0.01, 0.1, 1, 10, 100]),
                    #  "factor": tune.sample_from(
                    #      lambda spec: np.random.uniform(0.01, 0.99)),
                    #  "lr": tune.sample_from(
                    #      lambda spec: 10 ** -np.random.uniform(low=1, high=5)),
                    #  "momentum": tune.sample_from(
                    #      lambda spec: np.random.uniform(0, 0.99)),
                    #  "weight_decay": tune.sample_from(
                    #      lambda spec: np.random.choice(np.concatenate((10 ** -np.random.uniform(low=1, high=5, size=100), [0])))),
                },
                "local_dir": "/home/kn15263s/workspace/ray_results",
            }
        },
        resume=True,
        verbose=1,
        search_alg=algo1,
        # scheduler=sched
    )
