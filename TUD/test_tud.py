import sys
sys.path.append('.')
import argparse
from glob import glob
import os
import torch
import numpy as np
from csv import DictReader


def evaluate(path):
    train_acc = []
    test_acc = []
    test_std = []
    for p in glob(os.path.join(path, '*/results.csv')):
        with open(p, 'r') as f:
            reader = DictReader(f)
            dicts = [d for d in reader]
            train_acc.append([d['train_acc'] for d in dicts])
            test_acc.append([d['test_acc'] for d in dicts])
            test_std.append([d['test_std'] for d in dicts])

    min_epoch = np.min([len(a) for a in test_acc])
    test_acc = np.float32([a[:min_epoch] for a in test_acc])
    test_std = np.float32([a[:min_epoch] for a in test_std])
    train_acc = np.float32([a[:min_epoch] for a in train_acc])

    mean_test_acc = np.mean(test_acc, axis=0)
    test_imd = np.mean(test_std, axis=0)
    mean_train_acc = np.mean(train_acc, axis=0)
    test_cmd = np.std(test_acc, axis=0)
    train_cmd = np.std(train_acc, axis=0)

    max_idx = np.argmax(mean_test_acc)

    print(f'Test: {mean_test_acc[max_idx]:.5f} (+-{test_cmd[max_idx]:.5f}) (+-{test_imd[max_idx]:.5f})\n'
          f'Train: {mean_train_acc[max_idx]:.5f} (+-{train_cmd[max_idx]:.5f})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="path to the models")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for model_dir in glob(args.model_dir):
        print(f'Evaluating {model_dir}...')
        evaluate(model_dir)


if __name__ == '__main__':
    main()
