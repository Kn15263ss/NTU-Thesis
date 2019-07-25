import os
import json
import shutil
import random
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


file_path = '/home/willy-huang/workspace/research/with_pretrain/optimization_method/bayesian_results'

for f in os.listdir(file_path):
    data_folder = os.path.join(file_path, f)
    dataset = f[:f.rfind("_")]
    fig1, ax1 = plt.subplots()
    fig1, ax2 = plt.subplots()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Accuracy')
    # ax1.set_ylim(0.7, 0.95)

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 5)
    for f2 in os.listdir(data_folder):
        diff_file = os.path.join(data_folder, f2)
        if os.path.isfile(os.path.join(diff_file, "result.json")):
            filenum = f2[f2.rfind("_", 0, 11)+1:f2.rfind("_", 0, 15)]
            seed_table = np.array([["epoch", "accuracy", "loss"]])
            seed_table = np.asarray(
                open_json(
                    os.path.join(
                        diff_file, "result.json"), seed_table))
            np.savetxt(os.path.join(diff_file, "seed_table.csv"),
                       seed_table, delimiter=',', fmt="%s")
            with open(os.path.join(diff_file, 'seed_table.csv'), newline='') as csvfile:
                seed = np.genfromtxt(
                    csvfile, delimiter=',', dtype=np.float32,
                    skip_header=True)
                if filenum == '2':
                    l1, = ax1.plot(
                        seed[:, 0].T, seed[:, 1].T, c='xkcd:red',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)
                    l9, = ax2.plot(
                        seed[:, 0].T, seed[:, 2].T, c='xkcd:red',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)

                elif filenum == '6':
                    l2, = ax1.plot(
                        seed[:, 0].T, seed[:, 1].T, c='xkcd:purple',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)
                    l10, = ax2.plot(
                        seed[:, 0].T, seed[:, 2].T, c='xkcd:purple',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)

                elif filenum == '10':
                    l3, = ax1.plot(
                        seed[:, 0].T, seed[:, 1].T, c='xkcd:blue',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)
                    l11, = ax2.plot(
                        seed[:, 0].T, seed[:, 2].T, c='xkcd:blue',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)

                elif filenum == '14':
                    l4, = ax1.plot(
                        seed[:, 0].T, seed[:, 1].T, c='xkcd:orange',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)
                    l12, = ax2.plot(
                        seed[:, 0].T, seed[:, 2].T, c='xkcd:orange',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)

                elif filenum == '18':
                    l5, = ax1.plot(
                        seed[:, 0].T, seed[:, 1].T, c='xkcd:grey',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)
                    l13, = ax2.plot(
                        seed[:, 0].T, seed[:, 2].T, c='xkcd:grey',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)

                elif filenum == '22':
                    l6, = ax1.plot(
                        seed[:, 0].T, seed[:, 1].T, c='xkcd:gold',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)
                    l14, = ax2.plot(
                        seed[:, 0].T, seed[:, 2].T, c='xkcd:gold',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)

                elif filenum == '26':
                    l7, = ax1.plot(
                        seed[:, 0].T, seed[:, 1].T, c='xkcd:grape',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)
                    l15, = ax2.plot(
                        seed[:, 0].T, seed[:, 2].T, c='xkcd:grape',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)

                elif filenum == '30':
                    l8, = ax1.plot(
                        seed[:, 0].T, seed[:, 1].T, c='xkcd:green',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)
                    l16, = ax2.plot(
                        seed[:, 0].T, seed[:, 2].T, c='xkcd:green',
                        marker='o', ls='--', markersize=6,
                        markevery=[0, -1],
                        label=filenum)

    ax1.legend(
        [l1, l2, l3, l4, l5, l6, l7, l8, ],
        ('bayesian_2', 'bayesian_6', 'bayesian_10',
         'bayesian_14', 'bayesian_18', 'bayesian_22',
         'bayesian_26', 'bayesian_30'),
        loc='lower right', shadow=True)
    ax1.set_title(dataset)

    ax2.legend(
        [l9, l10, l11, l12, l13, l14, l15, l16, ],
        ('bayesian_2', 'bayesian_6', 'bayesian_10',
         'bayesian_14', 'bayesian_18', 'bayesian_22',
         'bayesian_26', 'bayesian_30'),
        loc='upper right', shadow=True)
    ax2.set_title(dataset)

    fig_path = '/home/willy-huang/Pictures/result/Bayesian/{}'.format(dataset)

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fig1.savefig(os.path.join(
        fig_path, '{}-{}-acc.png'.format(dataset, model)))
    fig1.savefig(os.path.join(
        fig_path, '{}-{}-acc.pdf'.format(dataset, model)))

    fig2.savefig(os.path.join(
        fig_path, '{}-{}-loss.png'.format(dataset, model)))
    fig2.savefig(os.path.join(
        fig_path, '{}-{}-loss.pdf'.format(dataset, model)))

    # plt.show()
