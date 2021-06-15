import sys
sys.path.append('.')
import torch
from torch.nn import CrossEntropyLoss
from torch_geometric.datasets import GNNBenchmarkDataset
from data_utils import preproc, CRaWlLoader
from models import CRaWl
from Benchmarks.train import train
import os
import argparse
import json
import numpy as np

DATA_NAME = 'CIFAR10'
DATA_PATH = f'data/{DATA_NAME}/'
PCKL_PATH = f'data/{DATA_NAME}/data.pckl'

num_node_feat, num_edge_feat, num_classes = 3, 1, 10


def load_split_data():
    if os.path.exists(PCKL_PATH):
        train_graphs, val_graphs, test_graphs = torch.load(PCKL_PATH)
    else:
        train_graphs = list(GNNBenchmarkDataset(DATA_PATH, DATA_NAME, split='train', transform=preproc))
        val_graphs = list(GNNBenchmarkDataset(DATA_PATH, DATA_NAME, split='val', transform=preproc))
        test_graphs = list(GNNBenchmarkDataset(DATA_PATH, DATA_NAME, split='test', transform=preproc))

        torch.save((train_graphs, val_graphs, test_graphs), PCKL_PATH)

    return train_graphs, val_graphs, test_graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/CIFAR10/default.json', help="path to config file")
    parser.add_argument("--name", type=str, default='0', help="path to config file")
    parser.add_argument("--gpu", type=int, default=0, help="id of gpu to be used for training")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.config, 'r') as f:
        config = json.load(f)

    model_dir = f'models/CIFAR10/{config["name"]}/{args.name}'

    train_graphs, val_graphs, _ = load_split_data()
    train_iter = CRaWlLoader(train_graphs, batch_size=config['batch_size'], num_workers=8, shuffle=True)
    val_iter = CRaWlLoader(val_graphs, batch_size=100, num_workers=8)

    model = CRaWl(model_dir, config, num_node_feat, num_edge_feat, num_classes, loss=CrossEntropyLoss())
    model.save()
    train(model, train_iter, val_iter, device=device)


if __name__ == '__main__':
    main()
