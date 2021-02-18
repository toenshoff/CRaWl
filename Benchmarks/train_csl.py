import sys
sys.path.append('.')
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from data_utils import preproc, CRaWlLoader
from models import CRaWl
from Benchmarks.train import train
from Benchmarks.test import test
import argparse
import json
import numpy as np

DATA_NAME = 'CSL'
DATA_PATH = f'data/{DATA_NAME}/'
PCKL_PATH = f'data/{DATA_NAME}/data.pckl'

num_node_feat, num_edge_feat, num_classes = 1, 1, 10


def get_idx(fold):
    with open(f'splits/CSL_Splits/train_idx-{fold + 1}.txt', 'r') as f:
        train_idx = [int(i) for i in f]
    with open(f'splits/CSL_Splits/val_idx-{fold + 1}.txt', 'r') as f:
        val_idx = [int(i) for i in f]
    with open(f'splits/CSL_Splits/test_idx-{fold + 1}.txt', 'r') as f:
        test_idx = [int(i) for i in f]

    return train_idx, val_idx, test_idx


def load_split_data(config, fold):
    graphs = GNNBenchmarkDataset(DATA_PATH, DATA_NAME, split='train', pre_transform=preproc).shuffle()
    train_idx, val_idx, test_idx = get_idx(fold)
    train_graphs, val_graphs, test_graphs = graphs[train_idx], graphs[val_idx], graphs[test_idx]

    train_iter = CRaWlLoader(train_graphs, batch_size=config['batch_size'])
    val_iter = CRaWlLoader(val_graphs, batch_size=config['batch_size'])
    test_iter = CRaWlLoader(test_graphs, batch_size=config['batch_size'])

    return train_iter, val_iter, test_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/CSL/default.json', help="path to config file")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps in each walk")
    parser.add_argument("--reps", type=int, default=10, help="Number of eval repetitions")
    parser.add_argument("--train", action='store_true', default=False, help="Train new models. If not set, pre-trained models will be evaluated on the test data")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)

    mean_list, std_list = [], []
    for f in range(5):
        train_iter, val_iter, test_iter = load_split_data(config, fold=f)

        model_dir = f'models/CSL/{config["name"]}/{f}'
        if args.train:
            model = CRaWl(model_dir, config, num_node_feat, num_edge_feat, num_classes)
            train(model, train_iter, val_iter)
        else:
            model = CRaWl.load(model_dir)

        mean, std = test(model, test_iter, repeats=args.reps, steps=args.steps)

        mean_list.append(mean)
        std_list.append(std)

    print(f'Mean Test Score {np.mean(mean_list)} (+-{np.std(mean_list)}), Mean STD {np.mean(std_list)}')


if __name__ == '__main__':
    main()
