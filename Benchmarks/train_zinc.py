import sys
sys.path.append('.')
import torch
import numpy as np
from torch_geometric.datasets import ZINC
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter
import json
import argparse
from data_utils import preproc, CRaWlLoader
from models import CRaWl


DATA_NAME = 'ZINC'
DATA_PATH = f'data/{DATA_NAME}/'
PCKL_PATH = f'data/{DATA_NAME}/data.pckl'

num_node_feat, num_edge_feat, num_classes = 21, 4, 1


def eval_regression(model, iter, repeats=1):
    model.eval()
    mae_list = []

    for _ in range(repeats):
        total_err = 0
        total_samples = 0
        for data in iter:
            data.to(device)
            data = model(data)

            err = torch.abs(data.y_pred - data.y).sum()
            total_err += err.cpu().detach().numpy()
            total_samples += data.y.shape[0]

        mae = total_err / float(total_samples)
        mae_list.append(mae)

    return np.mean(mae_list), np.std(mae_list)


def train_regression(model, train_iter, val_iter):
    writer = SummaryWriter(model.model_dir)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=model.config['lr'], weight_decay=model.config['weight_decay'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=model.config['decay_factor'], patience=model.config['patience'], verbose=True)

    best_val_mae = 1.0

    max_epochs = model.config['epochs']
    for e in range(1, max_epochs+1):
        train_loss = []
        total_train_mae = 0.0
        total_train_samples = 0

        model.train()

        for data in train_iter:
            data = data.to(device)

            opt.zero_grad()
            data = model(data)

            loss = model.loss(data)

            loss.backward()
            opt.step()

            mae = torch.abs(data.y_pred - data.y).sum()
            total_train_mae += mae.cpu().detach().numpy()
            total_train_samples += data.y.shape[0]
            train_loss.append(loss.cpu().detach().numpy())

        torch.cuda.empty_cache()

        train_mae = total_train_mae / float(total_train_samples)
        train_loss = np.mean(train_loss)
        val_mae, val_std = eval_regression(model, val_iter, model.config['eval_rep'])

        if val_mae <= best_val_mae:
            best_val_mae = val_mae
            model.save()

        print(f'Epoch {e} Loss: {train_loss:.4f}, Train: {train_mae:.4f}, Val: {val_mae:.4f} (+-{val_std:.4f})')

        writer.add_scalar('Loss/train', train_loss, e)
        writer.add_scalar('MAE/train', train_mae, e)
        writer.add_scalar('MAE/val', val_mae, e)

        sch.step(val_mae)
        if sch.state_dict()['_last_lr'][0] < 0.00001 or e > max_epochs:
            break


def feat_transform(graph):
    graph.x = torch._cast_Float(F.one_hot(graph.x.view(-1), num_node_feat))
    graph.edge_attr = torch._cast_Float(F.one_hot(graph.edge_attr.view(-1), num_edge_feat))
    return graph


def load_split_data(config):
    train_data = ZINC(DATA_PATH, subset=True, split='train', transform=feat_transform, pre_transform=preproc)
    val_data = ZINC(DATA_PATH, subset=True, split='val', transform=feat_transform, pre_transform=preproc)

    train_iter = CRaWlLoader(train_data, shuffle=True, batch_size=config['batch_size'], num_workers=4)
    val_iter = CRaWlLoader(val_data, batch_size=100, num_workers=4)
    return train_iter, val_iter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/ZINC/default.json', help="path to config file")
    parser.add_argument("--name", type=str, default='0', help="name of the model")
    parser.add_argument("--gpu", type=int, default=0, help="id of gpu to be used for training")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.config, 'r') as f:
        config = json.load(f)

    model_dir = f'models/ZINC/{config["name"]}/{args.name}'

    train_iter, val_iter = load_split_data(config)

    model = CRaWl(model_dir, config, num_node_feat, num_edge_feat, num_classes, L1Loss())
    model.save()
    train_regression(model, train_iter, val_iter)