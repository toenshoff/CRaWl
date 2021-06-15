import sys
sys.path.append('.')
from models import CRaWl
from data_utils import CRaWlLoader
from Benchmarks.test import test
import argparse
from glob import glob
import torch
import numpy as np

from Benchmarks.train_cifar import load_split_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="path to the model directories")
    parser.add_argument("--reps", type=int, default=10, help="Number of eval repetitions")
    parser.add_argument("--steps", type=int, default=150, help="Number of steps in each walk")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--split", type=str, default='test', choices={'test', 'val'}, help="split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size used for testing")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, val_graphs, test_graphs = load_split_data()
    iter = CRaWlLoader(test_graphs if args.split == 'test' else val_graphs, batch_size=args.batch_size, num_workers=8)

    mean_list, std_list = [], []
    model_list = sorted(list(glob(args.model_dir)))
    for model_dir in model_list:
        print(f'Evaluating {model_dir}...')
        model = CRaWl.load(model_dir)
        mean, std = test(model, iter, repeats=args.reps, steps=args.steps)

        mean_list.append(mean)
        std_list.append(std)

    print(f'Mean Score {np.mean(mean_list):.5f} (+-{np.std(mean_list):.5f} CMD) (+-{np.mean(std_list):.5f} IMD)')


if __name__ == '__main__':
    main()
