import os
import shutil
import argparse
import numpy as np

from distutils.dir_util import copy_tree, remove_tree


def check_file(local_path):

    if not os.path.exists(local_path):
        os.makedirs(local_path)


def main():

    parse = argparse.ArgumentParser()
    parse.add_argument('--file_path', type=str, default='')

    args, unparsed = parse.parse_known_args()

    # example:'/home/willy-huang/workspace/research/with_pretrain/gridesearch_results'
    file_path = args.file_path

    method = file_path[file_path.rfind('/')+1:file_path.rfind('_')]
    print(file_path)
    files = os.listdir(file_path)

    for f in files:  # f is a dataset of nn model. ex:CIFAR10.Vgg11
        path = os.path.join(file_path, f)
        best = []
        best_file = []
        # p is hyperparameter config. ex:HyperTrain_2_...
        for p in os.listdir(path):
            path2 = os.path.join(path, p, 'seed(50).csv')
            if os.path.isfile(path2):
                best_file.append(p)
                with open(path2, newline='') as csvfile:
                    seed = np.genfromtxt(csvfile, delimiter=',', dtype=str)
                    best.append(np.float32(seed[1][-2]))

        assert len(best) == len(best_file)
        best_path = os.path.join(path, best_file[best.index(min(best))])
        dataset = f[:f.rfind('.')]
        model = f[f.rfind('.')+1: f.rfind('_')]

        local_path = os.path.join(
            './best_loss_parameter', dataset, model, method)
        check_file(local_path)  # if local_file is not exist, it will make dir

        if not os.path.isdir(best_path):
            pass
        else:
            if not os.listdir(local_path) == []:
                remove_tree(local_path)
                check_file(local_path)
                copy_tree(best_path, local_path)
            else:
                copy_tree(best_path, local_path)

        best.clear()
        best_file.clear()

    print("\n Process Done!!! \n")


if __name__ == "__main__":
    main()
