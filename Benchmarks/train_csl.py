import sys
sys.path.append('.')
import torch
from torch.nn import CrossEntropyLoss
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.utils import to_networkx
from data_utils import preproc, CRaWlLoader
from models import CRaWl
from Benchmarks.train import train
from Benchmarks.test import test
import argparse
import json
import numpy as np
import scipy.sparse as sp
import dgl

DATA_NAME = 'CSL'
DATA_PATH = f'data/{DATA_NAME}/'
PCKL_PATH = f'data/{DATA_NAME}/data.pckl'


def get_idx(fold):
    with open(f'splits/CSL_Splits/train_idx-{fold + 1}.txt', 'r') as f:
        train_idx = [int(i) for i in f]
    with open(f'splits/CSL_Splits/val_idx-{fold + 1}.txt', 'r') as f:
        val_idx = [int(i) for i in f]
    with open(f'splits/CSL_Splits/test_idx-{fold + 1}.txt', 'r') as f:
        test_idx = [int(i) for i in f]

    return train_idx, val_idx, test_idx


def positional_encoding(pg_graph, pos_enc_dim=20):
    """
        Graph positional encoding v/ Laplacian eigenvectors (adapted from https://github.com/graphdeeplearning/benchmarking-gnns)
    """

    g = dgl.DGLGraph()
    g.from_networkx(nx_graph=to_networkx(pg_graph))

    n = g.number_of_nodes()
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N
    # Eigenvectors
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    pg_graph.x = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    return pg_graph


def load_split_data(config, fold, use_lap_feat=False):
    graphs = GNNBenchmarkDataset(root='data', name='CSL', split='train', pre_transform=preproc, use_node_attr=True).shuffle()
    train_idx, val_idx, test_idx = get_idx(fold)
    train_graphs, val_graphs, test_graphs = graphs[train_idx], graphs[val_idx], graphs[test_idx]

    if use_lap_feat:
        train_graphs = [positional_encoding(g) for g in train_graphs]
        val_graphs, [positional_encoding(g) for g in val_graphs]
        test_graphs = [positional_encoding(g) for g in test_graphs]

    train_iter = CRaWlLoader(train_graphs, batch_size=config['batch_size'])
    val_iter = CRaWlLoader(val_graphs, batch_size=config['batch_size'])
    test_iter = CRaWlLoader(test_graphs, batch_size=config['batch_size'])

    return train_iter, val_iter, test_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/CSL/default.json', help="path to config file")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--steps", type=int, default=150, help="Number of steps in each walk")
    parser.add_argument("--reps", type=int, default=10, help="Number of eval repetitions")
    parser.add_argument("--use_lap_feat", action='store_true', default=False, help="Whether or not to use the Laplacian positional encodings")
    parser.add_argument("--train", action='store_true', default=False, help="Train new models. If not set, pre-trained models will be evaluated on the test data")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_node_feat = 20 if args.use_lap_feat else 1
    num_edge_feat = 1
    num_classes = 10

    with open(args.config, 'r') as f:
        config = json.load(f)

    mean_list, std_list = [], []
    for f in range(5):
        train_iter, val_iter, test_iter = load_split_data(config, fold=f, use_lap_feat=args.use_lap_feat)

        model_dir = f'models/CSL/{config["name"]}/{f}'
        if args.train:
            model = CRaWl(model_dir, config, num_node_feat, num_edge_feat, num_classes, loss=CrossEntropyLoss())
            train(model, train_iter, val_iter)
        else:
            model = CRaWl.load(model_dir)

        mean, std = test(model, test_iter, repeats=args.reps, steps=args.steps)

        mean_list.append(mean)
        std_list.append(std)

    print(f'Mean Test Score {np.mean(mean_list):.5f} (+-{np.std(mean_list):.5f}), Mean STD {np.mean(std_list):.5f}')


if __name__ == '__main__':
    main()
