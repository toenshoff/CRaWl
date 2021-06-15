import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models import CRaWl
from OGB.ogb_utils import load_graphs, get_hiv_encoders, get_pcba_encoders
from data_utils import CRaWlLoader
from ogb.graphproppred.evaluate import Evaluator
import json
import os
import argparse

from tqdm import tqdm

score_keys = {'MOLHIV': 'rocauc', 'MOLPCBA': 'ap'}


def eval(model, iter, repeats=1):
    model.eval()

    scores = []
    for _ in range(repeats):
        all_y_pred = []
        all_y_true = []
        for data in iter:
            data = data.to(device)
            y_true = data.y
            data = model(data)

            y_pred = torch.sigmoid(data.y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            all_y_pred.append(y_pred)
            all_y_true.append(y_true)

        score = evaluator.eval({"y_true": np.vstack(all_y_true), "y_pred": np.vstack(all_y_pred)})
        scores.append(score[score_key])

    return np.mean(scores), np.std(scores)


def train(model, train_iter, val_iter):
    writer = SummaryWriter(model.model_dir)
    res_path = os.path.join(model.model_dir, 'results.json')

    model.to(device)

    decay_step = model.config['decay_step']
    opt = torch.optim.Adam(model.parameters(), lr=model.config['lr'], weight_decay=model.config['weight_decay'])
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, [decay_step], gamma=model.config['decay_factor'], verbose=True)

    best_val_score = 0.0

    walk_start_p = model.config['train_start_ratio']
    max_epochs = model.config['epochs']

    for e in range(1, max_epochs+1):
        train_loss = []
        all_y_pred = []
        all_y_true = []

        model.train()
        for data in tqdm(train_iter):
            opt.zero_grad()
            data = data.to(device)

            data = model(data, walk_start_p=walk_start_p)
            y_true = data.y

            no_nan = ~torch.isnan(y_true)
            loss = (F.binary_cross_entropy_with_logits(data.y_pred[no_nan], torch._cast_Float(y_true[no_nan])))

            loss.backward()
            opt.step()

            y_pred = torch.sigmoid(data.y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            all_y_pred.append(y_pred)
            all_y_true.append(y_true)
            train_loss.append(loss.cpu().detach().numpy())

        score = evaluator.eval({"y_true": np.vstack(all_y_true), "y_pred": np.vstack(all_y_pred)})
        train_score = score[score_key]
        train_loss = np.mean(train_loss)

        model.save(f'model_epoch_{e}')

        val_score, val_std = eval(model, val_iter, repeats=model.config['eval_rep'])
        print(f'Epoch {e} Loss: {train_loss:.4f}, Train: {train_score:.4f}, Val: {val_score:.4f} (+-{val_std:.4f})')

        if val_score >= best_val_score:
            best_val_score = val_score
            model.save()

        writer.add_scalar('Score/val', val_score, e)

        writer.add_scalar('Loss/train', train_loss, e)
        writer.add_scalar('Score/train', train_score, e)

        sch.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/MOLPCBA/default.json', help="path to config file")
    parser.add_argument("--data", type=str, default='MOLPCBA', choices={'MOLHIV', 'MOLPCBA'}, help="OGB Dataset to use")
    parser.add_argument("--name", type=str, default='0', help="path to config file")
    parser.add_argument("--gpu", type=int, default=0, help="id of gpu to be used for training")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scaler = torch.cuda.amp.GradScaler()

    with open(args.config, 'r') as f:
        config = json.load(f)

    ogb_data_name = 'ogbg-molhiv' if args.data == 'MOLHIV' else 'ogbg-molpcba'

    name = f'{args.name}_{config["name"]}'
    model_dir = f'models/{args.data}/{config["name"]}/{args.name}'
    print(model_dir)
    PCKL_PATH = f'data/OGB/{args.data}.pckl'

    score_key = score_keys[args.data]
    evaluator = Evaluator(ogb_data_name)

    if args.data == 'MOLHIV':
        AtomEncoder, BondEncoder = get_hiv_encoders(device)
    else:
        AtomEncoder, BondEncoder = get_pcba_encoders(device)
    num_node_feat = AtomEncoder.dim()
    num_edge_feat = BondEncoder.dim()

    print('Loading Graphs...')

    if os.path.exists(PCKL_PATH):
        out_dim, train_graphs, valid_graphs, _ = torch.load(PCKL_PATH)
    else:
        out_dim, train_graphs, valid_graphs, test_graphs = load_graphs(ogb_data_name)
        os.makedirs('data/OGB', exist_ok=True)
        torch.save((out_dim, train_graphs, valid_graphs, test_graphs), PCKL_PATH)

    train_iter = CRaWlLoader(train_graphs, batch_size=config['batch_size'], shuffle=True, num_workers=10)
    valid_iter = CRaWlLoader(valid_graphs, batch_size=150, num_workers=10)

    model = CRaWl(model_dir,
                  config,
                  num_node_feat,
                  num_edge_feat,
                  out_dim,
                  node_feat_enc=AtomEncoder,
                  edge_feat_enc=BondEncoder,
                  loss=CrossEntropyLoss())

    print('Training...')
    train(model, train_iter, valid_iter)

