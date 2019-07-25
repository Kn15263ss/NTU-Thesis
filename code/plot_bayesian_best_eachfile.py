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
    all_data_folder = os.path.join(file_path, f)
    dataset = f[:f.rfind(".")]
    model = f[f.rfind(".")+1:f.rfind("_")]

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Accuracy')
    # ax1.set_ylim(0.7, 0.95)

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 5)

    new_data_folder = []
    for f2 in os.listdir(all_data_folder):
        if os.path.isfile(os.path.join(all_data_folder, f2, 'seed(50).csv')):
            new_data_folder.append(f2)

    index_foder = []
    for i in new_data_folder:
        index_foder.append(
            (i, int(i[i.rfind("_", 0, 11)+1:i.rfind("_", 0, 15)])))
    sort_folder = sorted(index_foder, key=lambda x: x[1])
    best_nums = []
    # The best parameters before 5 configs
    for index in [2, 6, 10, 14, 18, 22, 26, 30]:
        all_data = []
        all_data_file = []
        for j in sort_folder[:index]:
            diff_file = os.path.join(all_data_folder, j[0])
            all_data_file.append(j[0])
            with open(os.path.join(diff_file, 'seed(50).csv'), newline='') as csvfile:
                seed = np.genfromtxt(csvfile, delimiter=',', dtype=str)
                all_data.append(np.float32(seed[1][-1]))

        assert len(all_data) == len(all_data_file)
        best_file = all_data_file[all_data.index(max(all_data))]
        best_path = os.path.join(all_data_folder, best_file)
        best_nums.append(best_file[best_file.rfind(
            "_", 0, 11)+1: best_file.rfind("_", 0, 15)])

        seed_table = np.array([["epoch", "accuracy", "loss"]])
        seed_table = np.asarray(
            open_json(
                os.path.join(
                    best_path, "result.json"), seed_table))
        np.savetxt(os.path.join(best_path, "seed_table.csv"),
                   seed_table, delimiter=',', fmt="%s")
        with open(os.path.join(best_path, 'seed_table.csv'), newline='') as csvfile:
            seed = np.genfromtxt(
                csvfile, delimiter=',', dtype=np.float32,
                skip_header=True)

            if str(index) == '2':
                l1, = ax1.plot(
                    seed[:, 0].T, seed[:, 1].T, c='xkcd:red',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))
                l9, = ax2.plot(
                    seed[:, 0].T, seed[:, 2].T, c='xkcd:red',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))

            elif str(index) == '6':
                l2, = ax1.plot(
                    seed[:, 0].T, seed[:, 1].T, c='xkcd:purple',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))
                l10, = ax2.plot(
                    seed[:, 0].T, seed[:, 2].T, c='xkcd:purple',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))

            elif str(index) == '10':
                l3, = ax1.plot(
                    seed[:, 0].T, seed[:, 1].T, c='xkcd:blue',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))
                l11, = ax2.plot(
                    seed[:, 0].T, seed[:, 2].T, c='xkcd:blue',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))

            elif str(index) == '14':
                l4, = ax1.plot(
                    seed[:, 0].T, seed[:, 1].T, c='xkcd:orange',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))
                l12, = ax2.plot(
                    seed[:, 0].T, seed[:, 2].T, c='xkcd:orange',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))

            elif str(index) == '18':
                l5, = ax1.plot(
                    seed[:, 0].T, seed[:, 1].T, c='xkcd:grey',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))
                l13, = ax2.plot(
                    seed[:, 0].T, seed[:, 2].T, c='xkcd:grey',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))

            elif str(index) == '22':
                l6, = ax1.plot(
                    seed[:, 0].T, seed[:, 1].T, c='xkcd:gold',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))
                l14, = ax2.plot(
                    seed[:, 0].T, seed[:, 2].T, c='xkcd:gold',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))

            elif str(index) == '26':
                l7, = ax1.plot(
                    seed[:, 0].T, seed[:, 1].T, c='xkcd:grape',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))
                l15, = ax2.plot(
                    seed[:, 0].T, seed[:, 2].T, c='xkcd:grape',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))

            elif str(index) == '30':
                l8, = ax1.plot(
                    seed[:, 0].T, seed[:, 1].T, c='xkcd:green',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))
                l16, = ax2.plot(
                    seed[:, 0].T, seed[:, 2].T, c='xkcd:green',
                    marker='o', ls='--', markersize=6,
                    markevery=[0, -1],
                    label=str(index))

    ax1.legend(
        [l1, l2, l3, l4, l5, l6, l7, l8, ],
        ('bay_{}'.format(best_nums[0]),
         'bay_{}'.format(best_nums[1]),
         'bay_{}'.format(best_nums[2]),
         'bay_{}'.format(best_nums[3]),
         'bay_{}'.format(best_nums[4]),
         'bay_{}'.format(best_nums[5]),
         'bay_{}'.format(best_nums[6]),
         'bay_{}'.format(best_nums[7])),
        loc='lower right', shadow=True)
    ax1.set_title(dataset+'.'+model)

    ax2.legend(
        [l9, l10, l11, l12, l13, l14, l15, l16, ],
        ('bay_{}'.format(best_nums[0]),
         'bay_{}'.format(best_nums[1]),
         'bay_{}'.format(best_nums[2]),
         'bay_{}'.format(best_nums[3]),
         'bay_{}'.format(best_nums[4]),
         'bay_{}'.format(best_nums[5]),
         'bay_{}'.format(best_nums[6]),
         'bay_{}'.format(best_nums[7])),
        loc='upper right', shadow=True)
    ax2.set_title(dataset+'.'+model)

    fig_path = '/home/willy-huang/Pictures/result/Bayesian/{}'.format(dataset)

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fig1.savefig(os.path.join(
        fig_path, '{}-{}-acc_best.png'.format(dataset, model)))
    fig1.savefig(os.path.join(
        fig_path, '{}-{}-acc_best.pdf'.format(dataset, model)))

    fig2.savefig(os.path.join(
        fig_path, '{}-{}-loss_best.png'.format(dataset, model)))
    fig2.savefig(os.path.join(
        fig_path, '{}-{}-loss_best.pdf'.format(dataset, model)))

    # plt.show()
