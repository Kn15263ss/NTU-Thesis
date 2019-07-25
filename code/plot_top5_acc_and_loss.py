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


file_path = '/home/willy-huang/workspace/research/top5_search/top5_parameter'
figs = []

for f in os.listdir(file_path):
    data_folder = os.path.join(file_path, f)
    if os.path.isdir(data_folder):
        dataset = f
        # if dataset == "CIFAR10":
        for m in os.listdir(data_folder):
            model = m
            model_folder = os.path.join(data_folder, m)
            # if model == "MobileNet_V2":
            fig1, ax1 = plt.subplots()
            fig1, ax2 = plt.subplots()

            ax1.set_xlabel('epoch')
            ax1.set_ylabel('Accuracy')
            # ax1.set_ylim(0.7, 0.95)

            ax2.set_xlabel('epoch')
            ax2.set_ylabel('Loss')
            ax2.set_ylim(0, 2.5)

            for d in os.listdir(model_folder):
                print(d)
                method = d
                all_table = []
                for i in range(1, 6):
                    all_path = os.path.join(
                        model_folder, d, "top{}".format(i))
                    seed_table = np.array(
                        [["epoch", "accuracy", "loss"]])
                    seed_table = np.asarray(
                        open_json(
                            os.path.join(
                                all_path, "result.json"), seed_table))
                    np.savetxt(
                        os.path.join(all_path, "seed_table.csv"),
                        seed_table, delimiter=',', fmt="%s")
                    with open(os.path.join(all_path, 'seed_table.csv'), newline='') as csvfile:
                        seed = np.genfromtxt(
                            csvfile, delimiter=',', dtype=np.float32,
                            skip_header=True)
                        all_table.append(seed)

                if method == "Hyperband_32":

                    t1_l1, = ax1.plot(
                        all_table[0][:, 0].T, all_table[0][:, 1].T,
                        c='xkcd:red', marker='o', ls='-', markersize=8,
                        markevery=[0, -1],
                        label=method)
                    t1_l2, = ax2.plot(
                        all_table[0][:, 0].T, all_table[0][:, 2].T,
                        c='xkcd:red', marker='o', ls='-', markersize=8,
                        markevery=[0, -1],
                        label=method)

                    t2_l1, = ax1.plot(
                        all_table[1][:, 0].T, all_table[1][:, 1].T,
                        c='xkcd:green', marker='o', ls='-',
                        markersize=8, markevery=[0, -1],
                        label=method)
                    t2_l2, = ax2.plot(
                        all_table[1][:, 0].T, all_table[1][:, 2].T,
                        c='xkcd:green', marker='o', ls='-',
                        markersize=8, markevery=[0, -1],
                        label=method)

                    t3_l1, = ax1.plot(
                        all_table[2][:, 0].T, all_table[2][:, 1].T,
                        c='xkcd:purple', marker='o', ls='-',
                        markersize=8, markevery=[0, -1],
                        label=method)
                    t3_l2, = ax2.plot(
                        all_table[2][:, 0].T, all_table[2][:, 2].T,
                        c='xkcd:purple', marker='o', ls='-',
                        markersize=8, markevery=[0, -1],
                        label=method)

                    t4_l1, = ax1.plot(
                        all_table[3][:, 0].T, all_table[3][:, 1].T,
                        c='xkcd:blue', marker='o', ls='-',
                        markersize=8, markevery=[0, -1],
                        label=method)
                    t4_l2, = ax2.plot(
                        all_table[3][:, 0].T, all_table[3][:, 2].T,
                        c='xkcd:blue', marker='o', ls='-',
                        markersize=8, markevery=[0, -1],
                        label=method)

                    t5_l1, = ax1.plot(
                        all_table[4][:, 0].T, all_table[4][:, 1].T,
                        c='xkcd:dark yellow', marker='o', ls='-',
                        markersize=8, markevery=[0, -1],
                        label=method)
                    t5_l2, = ax2.plot(
                        all_table[4][:, 0].T, all_table[4][:, 2].T,
                        c='xkcd:dark yellow', marker='o', ls='-',
                        markersize=8, markevery=[0, -1],
                        label=method)

                elif method == "randomsearch_32":

                    t1_l3, = ax1.plot(
                        all_table[0][:, 0].T, all_table[0][:, 1].T,
                        c='xkcd:red', marker='', ls=':', markersize=6,
                        markevery=[0, -1],
                        label=method)
                    t1_l4, = ax2.plot(
                        all_table[0][:, 0].T, all_table[0][:, 2].T,
                        c='xkcd:red', marker='', ls=':', markersize=6,
                        markevery=[0, -1],
                        label=method)

                    t2_l3, = ax1.plot(
                        all_table[1][:, 0].T, all_table[1][:, 1].T,
                        c='xkcd:green', marker='', ls=':',
                        markersize=6, markevery=[0, -1],
                        label=method)
                    t2_l4, = ax2.plot(
                        all_table[1][:, 0].T, all_table[1][:, 2].T,
                        c='xkcd:green', marker='', ls=':',
                        markersize=6, markevery=[0, -1],
                        label=method)

                    t3_l3, = ax1.plot(
                        all_table[2][:, 0].T, all_table[2][:, 1].T,
                        c='xkcd:purple', marker='', ls=':',
                        markersize=6, markevery=[0, -1],
                        label=method)
                    t3_l4, = ax2.plot(
                        all_table[2][:, 0].T, all_table[2][:, 2].T,
                        c='xkcd:purple', marker='', ls=':',
                        markersize=6, markevery=[0, -1],
                        label=method)

                    t4_l3, = ax1.plot(
                        all_table[3][:, 0].T, all_table[3][:, 1].T,
                        c='xkcd:blue', marker='', ls=':',
                        markersize=6, markevery=[0, -1],
                        label=method)
                    t4_l4, = ax2.plot(
                        all_table[3][:, 0].T, all_table[3][:, 2].T,
                        c='xkcd:blue', marker='', ls=':',
                        markersize=6, markevery=[0, -1],
                        label=method)

                    t5_l3, = ax1.plot(
                        all_table[4][:, 0].T, all_table[4][:, 1].T,
                        c='xkcd:dark yellow', marker='', ls=':',
                        markersize=6, markevery=[0, -1],
                        label=method)
                    t5_l4, = ax2.plot(
                        all_table[4][:, 0].T, all_table[4][:, 2].T,
                        c='xkcd:dark yellow', marker='', ls=':',
                        markersize=6, markevery=[0, -1],
                        label=method)

            ax1.legend(
                [t1_l1, t2_l1, t3_l1, t4_l1, t5_l1, t1_l3, t2_l3,
                    t3_l3, t4_l3, t5_l3, ],
                ('Hyperband_top1', 'Hyperband_top2', 'Hyperband_top3',
                    'Hyperband_top4', 'Hyperband_top5', 'random_top1',
                    'random_top2', 'random_top3', 'random_top4',
                    'random_top5'),
                loc='lower right', shadow=True)
            ax1.set_title(dataset+'-'+model)

            ax2.legend(
                [t1_l2, t2_l2, t3_l2, t4_l2, t5_l2, t1_l4, t2_l4, t3_l4, t4_l4, t5_l4, ],
                ('Hyperband_top1', 'Hyperband_top2', 'Hyperband_top3',
                    'Hyperband_top4', 'Hyperband_top5', 'random_top1',
                    'random_top2', 'random_top3', 'random_top4',
                    'random_top5'),
                loc='upper right', shadow=True)
            ax2.set_title(dataset+'-'+model)

            plt.show()
