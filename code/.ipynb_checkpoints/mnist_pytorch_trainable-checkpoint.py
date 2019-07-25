import os
import numpy as np
import time
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from PIL import Image
from glob import glob
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
        
def main():
    
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

    trainset_ld = DataLoader(dataset, batch_size=256, sampler=train_sampler, num_workers=4)
    validset_ld = DataLoader(dataset, batch_size=256, sampler=valid_sampler, num_workers=4)

    modelname = './{}.pth.tar'.format("simplenet")
    loggername = modelname.replace("pth.tar", "log")
    logger = utils.buildLogger(loggername)

    # ---- hyperparameters ----
    lr = 0.1
    momentum = 0.99
    weight_decay = 0.1
    factor = 0.999
    epoch = 10

    loss = F.nll_loss
    accuracy = -999999999999.0
    # -------------------- SETTINGS: NETWORK ARCHITECTURE
    model = simplenet().cuda()

    model = torch.nn.DataParallel(model).cuda()

    logger.info("Build Model Done")
    # -------------------- SETTINGS: OPTIMIZER & SCHEDULER --------------------
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                                  lr=lr,
                                  momentum=momentum,
                                  weight_decay=weight_decay,
                                  nesterov=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=factor,
                                                     patience=10, mode='min')

    logger.info("Build Optimizer Done")

    for epochID in range(0, epoch):
        
        model.train()

        for batchID, (input, target) in enumerate(trainset_ld):
            varInput = Variable(input).cuda(async=True)
            varTarget = Variable(target).cuda(async=True)
            varOutput = model(varInput)
            lossvalue = loss(varOutput, varTarget)
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()


        start_time = time.time()

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

        outLoss = lossVal / lossValNorm
        accuracy = correct.item() / num_samples

        scheduler.step(outLoss, epoch=epochID)

        if accuracy > accuracy:
            accuracy = accuracy

        logger.info('Epoch [' + str(epochID + 1) + '] [save] loss= {:.5f}'.format(outLoss) +
                             ' ---- accuracy= {:.5f}'.format(accuracy) +
                             ' ---- best_accuracy= {:.5f}'.format(accuracy) +
                             ' ---- model: {}'.format(modelname) +
                             ' ---- time: {:.1f} s'.format((time.time() - start_time)))

if __name__ == "__main__":
    main()
