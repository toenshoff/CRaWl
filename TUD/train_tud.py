import sys
sys.path.append('.')
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import numpy as np
from data_utils import preproc, CRaWlLoader
from models import CRaWl
from Benchmarks.train import eval
import argparse
import json
from csv import DictWriter
import os
from torch.utils.tensorboard import SummaryWriter

DATASETS = {'REDDIT-BINARY',
            'COLLAB',
            'IMDB-MULTI'}

NODE_FEAT = {'REDDIT-BINARY': 1,
            'COLLAB': 1,
            'IMDB-MULTI': 1}

EDGE_FEAT = {'REDDIT-BINARY': 1,
            'COLLAB': 1,
            'IMDB-MULTI': 1}

OUT_DIM = {'REDDIT-BINARY': 1,
           'COLLAB': 3,
           'IMDB-MULTI': 3}


def train(model, train_iter, val_iter):
    writer = SummaryWriter(model.model_dir)
    res_path = os.path.join(model.model_dir, 'results.csv')

    with open(res_path, 'w', newline='') as f:
        csv_writer = DictWriter(f, fieldnames=['train_acc', 'test_acc', 'test_std'])
        csv_writer.writeheader()

        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=model.config['lr'], weight_decay=model.config['weight_decay'])
        sch = torch.optim.lr_scheduler.StepLR(opt, gamma=model.config['decay_factor'], step_size=model.config['patience'])

        max_epochs = model.config['epochs']

        binary = model.out_dim == 1

        for e in range(1, max_epochs+1):
            train_acc = []
            train_loss = []

            model.train()
            for data in train_iter:
                data = data.to(device)
                y = data.y
                opt.zero_grad()
                data = model(data)
                logits = data.y_pred

                if binary:
                    y_pred = torch.sigmoid(logits)
                    loss = F.binary_cross_entropy(y_pred, torch._cast_Float(y), reduction='mean')
                    acc = torch._cast_Int(y_pred > 0.5).eq(y).sum() / float(y.shape[0])
                else:
                    loss = F.cross_entropy(logits, y.reshape(-1), reduction='mean')
                    acc = logits.argmax(dim=1).eq(y.reshape(-1)).sum() / float(y.shape[0])

                loss.backward()
                opt.step()

                train_acc.append(acc.cpu().detach().numpy())
                train_loss.append(loss.cpu().detach().numpy())

            torch.cuda.empty_cache()

            train_acc = np.mean(train_acc)
            train_loss = np.mean(train_loss)
            val_acc, val_std = eval(model, val_iter, repeats=model.config['eval_rep'])

            writer.add_scalar('Loss/train', train_loss, e)
            writer.add_scalar('Acc/train', train_acc, e)
            writer.add_scalar('Acc/val', val_acc, e)

            csv_writer.writerow({'train_acc': train_acc, 'test_acc': val_acc, 'test_std': val_std})

            print(f'Epoch {e + 1} Loss: {train_loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f} (+-{val_std:.4f})')
            sch.step()


def get_idx(dataset, split_dir, fold):
    with open(os.path.join(split_dir, dataset, f'train_idx-{fold + 1}.txt'), 'r') as f:
        train_idx = [int(i) for i in f]
    with open(os.path.join(split_dir, dataset, f'test_idx-{fold + 1}.txt'), 'r') as f:
        test_idx = [int(i) for i in f]
    return train_idx, test_idx


def get_split_data(data_name, split_dir, config, fold):
    dataset = TUDataset(f'data/TUD/{data_name}', data_name, transform=preproc)

    train_idx, test_idx = get_idx(data_name, split_dir, fold)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    train_iter = CRaWlLoader(list(dataset[train_idx]), batch_size=config['batch_size'], shuffle=True)
    val_iter = CRaWlLoader(list(dataset[test_idx]), batch_size=config['batch_size'] if 'eval_batch_size' not in config.keys() else config['eval_batch_size'])

    return train_iter, val_iter, val_iter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--name", type=str, default='0', help="path to config file")
    parser.add_argument("--data", type=str, help="Name of the dataset")
    parser.add_argument("--split_dir", type=str, default='data/TUD_Val_Splits', help="Path to the dataset splits")
    parser.add_argument("--gpu", type=int, default=0, help="id of gpu to be used for training")
    parser.add_argument("--fold", type=int, default=0, help="number of the fold to be used for testing")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.config, 'r') as f:
        config = json.load(f)

    model_dir = f'models/{args.data}/{config["name"]}/{args.name}'

    num_node_feat, num_edge_feat, num_classes = NODE_FEAT[args.data], EDGE_FEAT[args.data], OUT_DIM[args.data]

    train_iter, val_iter, _ = get_split_data(args.data, args.split_dir, config, args.fold)

    model = CRaWl(model_dir, config, num_node_feat, num_edge_feat, num_classes, loss=CrossEntropyLoss)
    model.save()
    train(model, train_iter, val_iter)

