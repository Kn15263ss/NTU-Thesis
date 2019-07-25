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
    parse.add_argument(
        '--file_path', type=str,
        default='/home/willy-huang/workspace/research/top5_search/randomsearch_32_results')

    args, unparsed = parse.parse_known_args()

    # example:'/home/willy-huang/workspace/research/with_pretrain/gridesearch_results'
    file_path = args.file_path

    method = file_path[file_path.rfind('/')+1:file_path.rfind('_')]
    print("search_top5_path: ", file_path)
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
                    best.append(np.float32(seed[1][-1]))

        assert len(best) == len(best_file)
        best_path = os.path.join(path, best_file[best.index(max(best))])
        dic = {}
        for i in range(len(best)):
            if best[i] not in dic.keys():
                dic[best[i]] = i

        top5_path = []
        top5 = []
        for i in range(1, 6):
            the_best = best_file[dic[sorted(dic)[-i]]]
            top5_path.append(os.path.join(
                path, the_best))
            temp = the_best[:the_best.rfind('_')]
            top5.append(temp[:temp.rfind('_')])
        dataset = f[:f.rfind('.')]
        model = f[f.rfind('.')+1: f.rfind('_')]

        for i in range(0, 5):

            local_path = os.path.join(
                './top5_search/top5_parameter', dataset, model, method,
                "top{}".format(i + 1))
            # if local_file is not exist, it will make dir
            check_file(local_path)

            if not os.path.isdir(top5_path[i]):
                pass
            else:
                if not os.listdir(local_path) == []:
                    remove_tree(local_path)
                    check_file(local_path)
                    copy_tree(top5_path[i], local_path)
                else:
                    copy_tree(top5_path[i], local_path)

            best.clear()
            best_file.clear()

    print("\n Process Done!!! \n")


if __name__ == "__main__":
    main()
