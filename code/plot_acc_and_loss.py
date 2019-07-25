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


file_path = '/home/willy-huang/workspace/research/with_pretrain/best_parameter'
figs = []

for f in os.listdir(file_path):
    data_folder = os.path.join(file_path, f)
    if os.path.isdir(data_folder):
        dataset = f
        # if dataset == "SVHN":
        for m in os.listdir(data_folder):
            model = m
            model_folder = os.path.join(data_folder, m)

            fig1, ax1 = plt.subplots(1)
            fig1, ax2 = plt.subplots(1)

            ax1.set_xlabel('epoch')
            ax1.set_ylabel('Accuracy')
            #ax1.set_ylim(0.7, 0.95)

            ax2.set_xlabel('epoch')
            ax2.set_ylabel('Loss')
            ax2.set_ylim(0, 2.5)

            ### hide the axis ###
            # ax1.get_xaxis().set_visible(False)
            # ax1.get_yaxis().set_visible(False)
            # ax2.get_xaxis().set_visible(False)
            # ax2.get_yaxis().set_visible(False)
            for d in os.listdir(model_folder):
                method = d
                all_path = os.path.join(
                    model_folder, d, "valid")
                seed_table = np.array([["epoch", "accuracy", "loss"]])
                seed_table = np.asarray(
                    open_json(
                        os.path.join(
                            all_path, "result.json"), seed_table))
                np.savetxt(os.path.join(all_path, "seed_table.csv"),
                           seed_table, delimiter=',', fmt="%s")
                with open(os.path.join(all_path, 'seed_table.csv'), newline='') as csvfile:
                    seed = np.genfromtxt(
                        csvfile, delimiter=',', dtype=np.float32,
                        skip_header=True)
                    #seed = np.array([[0, 0, 0]])

                    if method == "gridsearch":
                        l1, = ax1.plot(
                            seed[:, 0].T, seed[:, 1].T, c='xkcd:black',
                            marker='o', ls='-', markersize=6,
                            markevery=[0, -1],
                            label=method)
                        l10, = ax2.plot(
                            seed[:, 0].T, seed[:, 2].T, c='xkcd:black',
                            marker='o', ls='-', markersize=6,
                            markevery=[0, -1],
                            label=method)

                    elif method == "randomsearch_8":
                        l2, = ax1.plot(
                            seed[:, 0].T, seed[:, 1].T, c='xkcd:baby blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[0, -1],
                            label=method)
                        l11, = ax2.plot(
                            seed[:, 0].T, seed[:, 2].T, c='xkcd:baby blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[0, -1],
                            label=method)

                    elif method == "randomsearch_16":
                        l3, = ax1.plot(
                            seed[:, 0].T, seed[:, 1].T, c='xkcd:blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[0, -1],
                            label=method)
                        l12, = ax2.plot(
                            seed[:, 0].T, seed[:, 2].T, c='xkcd:blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[0, -1],
                            label=method)

                    elif method == "randomsearch_32":
                        l4, = ax1.plot(
                            seed[:, 0].T, seed[:, 1].T, c='xkcd:grey blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[0, -1],
                            label=method)
                        l13, = ax2.plot(
                            seed[:, 0].T, seed[:, 2].T, c='xkcd:grey blue',
                            marker='o', ls='-.', markersize=6,
                            markevery=[0, -1],
                            label=method)

                    elif method == "bayesian":
                        l5, = ax1.plot(
                            seed[:, 0].T, seed[:, 1].T, c='xkcd:green',
                            marker='o', ls='--', markersize=6,
                            markevery=[0, -1],
                            label=method)
                        l14, = ax2.plot(
                            seed[:, 0].T, seed[:, 2].T, c='xkcd:green',
                            marker='o', ls='--', markersize=6,
                            markevery=[0, -1],
                            label=method)

                    elif method == "Hyperband_32":
                        l6, = ax1.plot(
                            seed[:, 0].T, seed[:, 1].T, c='xkcd:red',
                            marker='o', ls=':', markersize=6,
                            markevery=[0, -1],
                            label=method)
                        l15, = ax2.plot(
                            seed[:, 0].T, seed[:, 2].T, c='xkcd:red',
                            marker='o', ls=':', markersize=6,
                            markevery=[0, -1],
                            label=method)

                    elif method == "Hyperband_64":
                        l7, = ax1.plot(
                            seed[:, 0].T, seed[:, 1].T, c='xkcd:pink',
                            marker='o', ls=':', markersize=6,
                            markevery=[0, -1],
                            label=method)
                        l16, = ax2.plot(
                            seed[:, 0].T, seed[:, 2].T, c='xkcd:pink',
                            marker='o', ls=':', markersize=6,
                            markevery=[0, -1],
                            label=method)

                    elif method == "Hyperband_128":
                        l8, = ax1.plot(
                            seed[:, 0].T, seed[:, 1].T, c='xkcd:orange',
                            marker='o', ls=':', markersize=6,
                            markevery=[0, -1],
                            label=method)
                        l17, = ax2.plot(
                            seed[:, 0].T, seed[:, 2].T, c='xkcd:orange',
                            marker='o', ls=':', markersize=6,
                            markevery=[0, -1],
                            label=method)

                    elif method == "Hyperband_256":
                        l9, = ax1.plot(
                            seed[:, 0].T, seed[:, 1].T,
                            c='xkcd:maroon', marker='o', ls=':',
                            markersize=6, markevery=[0, -1],
                            label=method)
                        l18, = ax2.plot(
                            seed[:, 0].T, seed[:, 2].T,
                            c='xkcd:maroon', marker='o', ls=':',
                            markersize=6, markevery=[0, -1],
                            label=method)

            # ax1.legend(
            #     [l1, l2, l3, l4, l5, l6, l7, l8, l9, ],
            #     ('gridsearch', 'randomsearch_8', 'randomsearch_16',
            #      'randomsearch_32', 'bayesian', 'Hyperband_32', 'Hyperband_64',
            #      'Hyperband_128', 'Hyperband_256'),
            #     loc='center', shadow=True, fontsize='xx-large', mode='expand')
            ax1.set_title(dataset+'-'+model)

            # ax2.legend(
            #     [l10, l11, l12, l13, l14, l15, l16, l17, l18, ],
            #     ('gridsearch', 'randomsearch_8', 'randomsearch_16',
            #      'randomsearch_32', 'bayesian', 'Hyperband_32', 'Hyperband_64',
            #      'Hyperband_128', 'Hyperband_256'),
            #     loc='center', shadow=True, fontsize='xx-large', mode='expand')
            ax2.set_title(dataset+'-'+model)

            fig_path = '/home/willy-huang/Pictures/result/{}'.format(
                dataset)

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
