import sys
sys.path.append('.')
import torch
from torch_geometric.datasets import ZINC
import numpy as np
from data_utils import preproc, CRaWlLoader
from models import CRaWl
import json
import os
from glob import glob
import argparse
from Benchmarks.train_zinc import DATA_PATH, feat_transform


def test(model, iter, repeats, steps):
    res_path = os.path.join(model.model_dir, 'test_results.json')

    model.to(device)
    model.eval()

    mae_list = []
    for _ in range(repeats):
        total_err = 0
        total_samples = 0
        for data in iter:
            data.to(device)
            y = data.y
            data = model(data, steps)
            err = torch.abs(data.y_pred - y).sum()
            total_err += err.cpu().detach().numpy()
            total_samples += y.shape[0]

            del data

        mae = total_err / float(total_samples)
        mae_list.append(mae)

    score = np.mean(mae_list)
    std = np.std(mae_list)

    print(f'Test Score: {score:.4f} (+-{std:.4f})')

    with open(res_path, 'w') as f:
        json.dump({'score': float(score), 'std': float(std)}, f, indent=4)

    return score, std


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="path to the model directories")
    parser.add_argument("--reps", type=int, default=10, help="Number of eval repetitions")
    parser.add_argument("--steps", type=int, default=150, help="Number of steps in each walk")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--split", type=str, default='test', choices={'test', 'val'}, help="split to evaluate on")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print('Loading Graphs...')

    data = ZINC(DATA_PATH, subset=True, split=args.split, transform=feat_transform, pre_transform=preproc)
    iter = CRaWlLoader(data, batch_size=50, num_workers=4)

    mean_list, std_list = [], []
    model_list = sorted(list(glob(args.model_dir)))
    for model_dir in model_list:
        print(f'Evaluating {model_dir}...')
        model = CRaWl.load(model_dir)
        mean, std = test(model, iter, repeats=args.reps, steps=args.steps)

        mean_list.append(mean)
        std_list.append(std)

    print(f'Mean Score {np.mean(mean_list):.5f} (+-{np.std(mean_list):.5f} CMD) (+-{np.mean(std_list):.5f} IMD)')