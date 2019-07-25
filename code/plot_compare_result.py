import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt

file_path = '/home/willy-huang/workspace/research/with_pretrain/test_nn'
figs = []

for r, d, f in sorted(os.walk(file_path)):

    if f == []:
        pass

    else:
        with open(os.path.join(r, 'seed(50).csv'), newline='') as csvfile:
            seed = np.genfromtxt(csvfile, delimiter=',',
                                 dtype=np.float32, skip_header=True)
            figs.append(seed)

        if len(figs) == 2:
            fig1 = np.asarray(figs[0]).T
            fig2 = np.asarray(figs[1]).T

            fig, ax1 = plt.subplots()

            ax1.set_xlabel('epoch')
            ax1.set_ylabel('Accuracy')

            # training accuracy
            l1, = ax1.plot(
                fig1[0],
                c='xkcd:red', marker='', ls='-', markersize=6,
                markevery=0.1)
            # training accuracy with weight_decay
            l2, = ax1.plot(fig2[0], c='xkcd:bright light blue',
                           marker='', ls='-', markersize=6, markevery=0.1)
            l3, = ax1.plot(fig1[2], c='xkcd:maroon',
                           marker='', ls='-', markersize=6,
                           markevery=0.1)  # valid accuracy
            # valid accuracy with weight_decay
            l4, = ax1.plot(fig2[2], c='xkcd:blue',
                           marker='', ls='-', markersize=6, markevery=0.1)
            ax1.tick_params(axis='y', )

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            # we already handled the x-label with ax1
            ax2.set_ylabel('Loss')
            l5, = ax2.plot(fig1[1], c='xkcd:red',
                           marker='', ls=':', markersize=6,
                           markevery=0.1)  # training loss
            # training loss with weight_decay
            l6, = ax2.plot(fig2[1], c='xkcd:bright light blue',
                           marker='', ls=':', markersize=6,
                           markevery=0.1)
            l7, = ax2.plot(fig1[3], c="xkcd:maroon",
                           marker='', ls=':', markersize=6,
                           markevery=0.1)  # vaild loss
            # valid loss with weight_decay
            l8, = ax2.plot(fig2[3], c='xkcd:blue',
                           marker='', ls=':', markersize=6,
                           markevery=0.1)
            ax2.tick_params(axis='y')

            ###--- style 1 ---###
            # ax1.set_xlim(-1, 70)
            # ax2.set_xlim(-1, 70)
            # first_legend = ax2.legend(
            #     [l1, l2, l3, l4, ],
            #     ('tr_acc', 'tr_acc_wd', 'val_acc', 'val_acc_wd'),
            #     loc='upper right', shadow=True)
            # plt.gca().add_artist(first_legend)
            # ax2.legend(
            #     [l5, l6, l7, l8, ],
            #     ('tr_loss', 'tr_loss_wd', 'va_loss', 'val_loss_wd'),
            #     loc='lower right', shadow=True)

            ###--- style 2 ---###
            # ax1.set_xlim(-1, 70)
            # ax2.set_xlim(-1, 70)
            # ax2.legend(
            #     [l1, l2, l3, l4, l5, l6, l7, l8, ],
            #     ('tr_acc', 'tr_acc_wd', 'val_acc', 'val_acc_wd', 'tr_loss',
            #      'tr_loss_wd', 'va_loss', 'val_loss_wd'),
            #     loc='Upper right', shadow=True)

            ###--- style 3 ---###
            # ax1.set_xlim(-25, 75)
            # ax2.set_xlim(-25, 75)
            # first_legend = ax2.legend(
            #     [l1, l2, l3, l4, ],
            #     ('tr_acc', 'tr_acc_wd', 'val_acc', 'val_acc_wd'),
            #     loc='upper left', shadow=True)
            # plt.gca().add_artist(first_legend)
            # ax2.legend(
            #     [l5, l6, l7, l8, ],
            #     ('tr_loss', 'tr_loss_wd', 'va_loss', 'val_loss_wd'),
            #     loc='lower right', shadow=True)

            ###--- style 4 ---###
            # ax1.set_xlim(-25, 75)
            # ax2.set_xlim(-25, 75)
            # first_legend = ax2.legend(
            #     [l1, l2, l3, l4, ],
            #     ('tr_acc', 'tr_acc_wd', 'val_acc', 'val_acc_wd'),
            #     loc='upper left', shadow=True)
            # plt.gca().add_artist(first_legend)
            # ax2.legend(
            #     [l5, l6, l7, l8, ],
            #     ('tr_loss', 'tr_loss_wd', 'va_loss', 'val_loss_wd'),
            #     loc='upper right', shadow=True)

            ###--- style 5 ---###
            # ax1.set_xlim(-25, 50)
            # ax2.set_xlim(-25, 50)
            # first_legend = ax2.legend(
            #     [l1, l2, l3, l4, ],
            #     ('tr_acc', 'tr_acc_wd', 'val_acc', 'val_acc_wd'),
            #     loc='upper left', shadow=True)
            # plt.gca().add_artist(first_legend)
            # ax2.legend(
            #     [l5, l6, l7, l8, ],
            #     ('tr_loss', 'tr_loss_wd', 'va_loss', 'val_loss_wd'),
            #     loc='lower left', shadow=True)

            ###--- style 6 ---###
            ax1.set_xlim(-1, 50)
            ax2.set_xlim(-1, 50)
            ax2.legend(
                [l1, l2, l3, l4, l5, l6, l7, l8, ],
                ('tr_acc', 'tr_acc_wd', 'val_acc', 'val_acc_wd', 'tr_loss',
                 'tr_loss_wd', 'va_loss', 'val_loss_wd'),
                loc='lower right', shadow=True)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.title(r[r.rfind('/')+1: r.rfind('--')])
            plt.subplots_adjust(top=0.9)
            plt.show()
            figs.clear()
