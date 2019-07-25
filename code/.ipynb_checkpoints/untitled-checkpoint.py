import os
import numpy as np
import time
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.suggest import BayesOptSearch, HyperOptSearch
from hyperopt import hp


from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

class simplenet(nn.Module):
    
        def __init__(self):
            super(simplenet, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

class TrainMNIST(Trainable):
    
    def _setup(self, config):
        torch.cuda.manual_seed(1)

        data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))])

        dataset = torchvision.datasets.MNIST('~/data', train=True,
                                             transform=data_transforms)

        num_total = len(dataset)
        shuffle = np.random.permutation(num_total)
        split_val = int(num_total * 0.2)

        train_idx, valid_idx = shuffle[split_val:], shuffle[:split_val]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.trainset_ld = DataLoader(dataset, batch_size=256, sampler=train_sampler, num_workers=4)
        self.validset_ld = DataLoader(dataset, batch_size=256, sampler=valid_sampler, num_workers=4)

        self.modelname = './{}.pth.tar'.format("simplenet")
        loggername = self.modelname.replace("pth.tar", "log")
        self.logger = utils.buildLogger(loggername)

        # ---- hyperparameters ----
        lr = config["lr"]
        momentum = config["momentum"]
        weight_decay = config["weight_decay"]
        factor = config["factor"]
        
        self.epochID = 0
        self.loss = F.nll_loss
        self.accuracy = -999999999999.0
        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        self.model = simplenet().cuda()

        #self.model = torch.nn.DataParallel(self.model).cuda()

        self.logger.info("Build Model Done")
        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER --------------------
        self.optimizer = optim.SGD(filter(lambda x: x.requires_grad, self.model.parameters()),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay,
                              nesterov=False)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              factor=factor,
                                                              patience=10, mode='min')

        self.logger.info("Build Optimizer Done")

    def _train_iteration(self):
        
        self.model.train()

        for batchID, (input, target) in enumerate(self.trainset_ld):
            varInput = Variable(input).cuda(async=True)
            varTarget = Variable(target).cuda(async=True)
            varOutput = self.model(varInput)

            lossvalue = self.loss(varOutput, varTarget)
            self.optimizer.zero_grad()
            lossvalue.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
            self.optimizer.step()

    def _test(self):
        start_time = time.time()

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
        
        outLoss = lossVal / lossValNorm
        accuracy = correct.item() / num_samples
        
        self.scheduler.step(outLoss, epoch=self.epochID)

        if accuracy > self.accuracy:
            self.accuracy = accuracy

        self.logger.info('Epoch [' + str(self.epochID + 1) + '] [save] loss= {:.5f}'.format(outLoss) +
                         ' ---- accuracy= {:.5f}'.format(accuracy) +
                         ' ---- best_accuracy= {:.5f}'.format(self.accuracy) +
                         ' ---- model: {}'.format(self.modelname) +
                         ' ---- time: {:.1f} s'.format((time.time() - start_time)))
            
        self.epochID+=1
        return {"neg_mean_loss": outLoss, "mean_accuracy": accuracy, "epoch": self.epochID}

    def _train(self):
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save({
            "accuracy": self.accuracy,
            "state_dict": self.model.state_dict(),
        }, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


if __name__ == "__main__":
    
    ray.init()
    
    sched = HyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        max_t=5
        )
    
    space1 = {
        "factor": (0.01, 0.999),
        "lr": (1, 5),
        "momentum": (0, 0.99),
        "weight_decay": (1, 5)
    }
    
    space2 = {
        'factor': hp.uniform('factor', 0.01, 0.999),
        'lr': 10 ** -hp.uniform('lr', 1, 5),
        'momentum': hp.uniform("momentum", 0, 0.99),
        'weight_decay': hp.choice("weight_decay", np.concatenate((10 ** -np.random.uniform(1, 5, size=100),[0])))
    }
    
    current_best_params = [
        {
            "factor": 0.1,
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0
        }
    ]
    
    algo2 = HyperOptSearch(
        space2,
        max_concurrent=4,
        reward_attr="neg_mean_loss",
        points_to_evaluate=current_best_params
    )
    
    algo1 = BayesOptSearch(
        space1,
        max_concurrent=10,
        reward_attr="neg_mean_loss",
        #reward_attr="mean_accuracy",
        utility_kwargs={
#             "kind": "ucb",
#             "kappa": 2.5,
#             "xi": 0.0
            "kind": "ei",
            "kappa": 2.5,
            "xi": 0.01
        },
        random_state="special"
    )
    
    tune.run_experiments(
        {
            "exp": {
                "stop": {
                    #"mean_accuracy": 0.98,
                    #"training_iteration": 100
                    "epoch": 10
                },
                "resources_per_trial": {
                    "cpu": 4,
                    "gpu": 1
                },
                "run": TrainMNIST,
                "max_failures":5,
                "checkpoint_at_end": True,
                "num_samples":20,
                "config": {
#                     "lr": tune.grid_search([0.01, 0.1, 0.5]),
#                     "momentum": tune.grid_search([0, 0.5, 0.8]),
                     "lr": tune.sample_from(
                         lambda spec: 10 ** np.random.uniform(-1, -5)),
                     "momentum": tune.sample_from(
                         lambda spec: np.random.uniform(0, 0.99))

                }
            }
        },
        verbose = 1,
        search_alg=algo2
#         scheduler=sched
    )
