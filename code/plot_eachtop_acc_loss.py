import os
import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt


def open_json(path, seed_table):
    table = seed_table
    with open(path, 'r') as f:
        count = 0
        for line in f.readlines():
            dic = json.loads(line)
            if "episode_reward_mean" in dic.keys():
                if dic["epoch"] == 1:
                    count += 1
                if count >= 2:
                    table = seed_table
                    count = 0
                table = np.append(
                    table,
                    [[dic["epoch"],
                        dic["mean_accuracy"],
                        dic["episode_reward_mean"]]],
                    axis=0)
            elif "mean_loss" in dic.keys():
                if dic["epoch"] == 1:
                    count += 1
                if count >= 2:
                    table = seed_table
                    count = 0
                table = np.append(
                    table,
                    [[dic["epoch"],
                        dic["mean_accuracy"],
                        dic["mean_loss"]]],
                    axis=0)
    return table


def all_parameter(datasetandmodel_folder, all_seed):
    for f3 in os.listdir(datasetandmodel_folder):
        file3 = os.path.join(
            datasetandmodel_folder, f3, 'result.json')
        if os.path.isfile(file3):
            seed_table = np.array([["epoch", "accuracy", "loss"]])
            seed_table = np.asarray(open_json(file3, seed_table))
            new_file3 = file3.replace(file3[file3.rfind("/"):], "")
            np.savetxt(os.path.join(new_file3, "seed_table.csv"),
                       seed_table, delimiter=',', fmt="%s")
            with open(os.path.join(new_file3, 'seed_table.csv'), newline='') as csvfile:
                seed = np.genfromtxt(
                    csvfile, delimiter=',', dtype=np.float32,
                    skip_header=True)
                if len(seed) < 50:
                    if len(seed.shape) == 1:
                        for j in range(2, 51):
                            seed = np.append(seed, [j, 0, 999])
                    elif len(seed.shape) == 2:
                        for j in range(len(seed)+1, 51):
                            seed = np.append(seed, [j, 0, 999])
                    seed = seed.reshape(50, 3)
                all_seed.append(seed)
    return all_seed


file_path = '/home/willy-huang/workspace/research/with_pretrain/optimization_method'
dataset = [
    "CIFAR10.Vgg11", "CIFAR10.Resnet18", "CIFAR10.MobileNet_V2",
    "FashionMNIST.Vgg11", "FashionMNIST.Resnet18", "FashionMNIST.MobileNet_V2",
    "KMNIST.Vgg11", "KMNIST.Resnet18", "KMNIST.MobileNet_V2", "SVHN.Vgg11",
    "SVHN.Resnet18", "SVHN.MobileNet_V2", "STL10.Vgg11", "STL10.Resnet18",
    "STL10.MobileNet_V2"]
for d in dataset:
    print(d)
    fig1, ax1 = plt.subplots()
    fig1, ax2 = plt.subplots()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Accuracy')
    # ax1.set_ylim(0.7, 0.95)

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 2.5)
    for f in os.listdir(file_path):
        data_folder = os.path.join(file_path, f)
        method = f[:f.rfind("_")]
        if method != "bayesian":
            for f2 in os.listdir(data_folder):
                datasetandmodel_folder = os.path.join(data_folder, f2)
                datasetandmodel = f2[:f2.rfind("_")]
                all_seed = []
                all_seed = np.asarray(all_parameter(
                    datasetandmodel_folder, all_seed))
                new_seed_acc = []
                new_seed_loss = []
                epoch = list(range(1, 51))
                if datasetandmodel == d:
                    for i in range(0, 50):
                        # calculate max accuracy of whole hp_configs on each epoch
                        new_seed_acc.append(max(all_seed[:, i, 1]))
                        new_seed_loss.append(min(all_seed[:, i, 2]))
                    if method == "gridsearch":
                        l1, = ax1.plot(
                            epoch, new_seed_acc, c='xkcd:black',
                            marker='o', ls='-', markersize=6,
                            markevery=[new_seed_acc.index(max(new_seed_acc))],
                            label=method)

                        l10, = ax2.plot(
                            epoch, new_seed_loss, c='xkcd:black',
                            marker='o', ls='-', markersize=6,
                            markevery=[new_seed_loss.index(min(new_seed_loss))],
                            label=method)

                    elif method == "randomsearch_8":
                        l2, = ax1.plot(
                            epoch, new_seed_acc, c='xkcd:baby blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[new_seed_acc.index(max(new_seed_acc))],
                            label=method)
                        l11, = ax2.plot(
                            epoch, new_seed_loss, c='xkcd:baby blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[new_seed_loss.index(min(new_seed_loss))],
                            label=method)

                    elif method == "randomsearch_16":
                        l3, = ax1.plot(
                            epoch, new_seed_acc, c='xkcd:blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[new_seed_acc.index(max(new_seed_acc))],
                            label=method)
                        l12, = ax2.plot(
                            epoch, new_seed_loss, c='xkcd:blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[new_seed_loss.index(min(new_seed_loss))],
                            label=method)

                    elif method == "randomsearch_32":
                        l4, = ax1.plot(
                            epoch, new_seed_acc, c='xkcd:grey blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[new_seed_acc.index(max(new_seed_acc))],
                            label=method)
                        l13, = ax2.plot(
                            epoch, new_seed_loss, c='xkcd:grey blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[new_seed_loss.index(min(new_seed_loss))],
                            label=method)

                    elif method == "Hyperband_32":
                        l5, = ax1.plot(
                            epoch, new_seed_acc, c='xkcd:red',
                            marker='o', ls=':', markersize=6,
                            markevery=[new_seed_acc.index(max(new_seed_acc))],
                            label=method)
                        l14, = ax2.plot(
                            epoch, new_seed_loss, c='xkcd:red',
                            marker='o', ls=':', markersize=6,
                            markevery=[new_seed_loss.index(min(new_seed_loss))],
                            label=method)

                    elif method == "Hyperband_64":
                        l6, = ax1.plot(
                            epoch, new_seed_acc, c='xkcd:pink',
                            marker='o', ls=':', markersize=6,
                            markevery=[new_seed_acc.index(max(new_seed_acc))],
                            label=method)
                        l15, = ax2.plot(
                            epoch, new_seed_loss, c='xkcd:pink',
                            marker='o', ls=':', markersize=6,
                            markevery=[new_seed_loss.index(min(new_seed_loss))],
                            label=method)

                    elif method == "Hyperband_128":
                        l7, = ax1.plot(
                            epoch, new_seed_acc, c='xkcd:orange',
                            marker='o', ls=':', markersize=6,
                            markevery=[new_seed_acc.index(max(new_seed_acc))],
                            label=method)
                        l16, = ax2.plot(
                            epoch, new_seed_loss, c='xkcd:orange',
                            marker='o', ls=':', markersize=6,
                            markevery=[new_seed_loss.index(min(new_seed_loss))],
                            label=method)

                    elif method == "Hyperband_256":
                        l8, = ax1.plot(
                            epoch, new_seed_acc,
                            c='xkcd:maroon', marker='o', ls=':',
                            markersize=6, markevery=[new_seed_acc.index(max(new_seed_acc))],
                            label=method)
                        l17, = ax2.plot(
                            epoch, new_seed_loss,
                            c='xkcd:maroon', marker='o', ls=':',
                            markersize=6, markevery=[new_seed_loss.index(min(new_seed_loss))],
                            label=method)

    ax1.legend(
        [l1, l2, l3, l4, l5, l6, l7, l8, ],
        ('gridsearch', 'randomsearch_8', 'randomsearch_16',
         'randomsearch_32', 'Hyperband_32', 'Hyperband_64',
         'Hyperband_128', 'Hyperband_256'),
        loc='lower right', shadow=True)
    ax1.set_title(d)

    ax2.legend(
        [l10, l11, l12, l13, l14, l15, l16, l17, ],
        ('gridsearch', 'randomsearch_8', 'randomsearch_16',
         'randomsearch_32', 'Hyperband_32', 'Hyperband_64',
         'Hyperband_128', 'Hyperband_256'),
        loc='upper right', shadow=True)
    ax2.set_title(d)
    plt.show()
    plt.clf()
