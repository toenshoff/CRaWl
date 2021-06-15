import sys
sys.path.append('.')
import torch
import numpy as np
from models import CRaWl
from data_utils import CRaWlLoader
from ogb.graphproppred.evaluate import Evaluator
import json
import os
from glob import glob
import argparse

score_keys = {'MOLHIV': 'rocauc', 'MOLPCBA': 'ap'}


def test(model, iter, repeats, steps=50):
    res_path = os.path.join(model.model_dir, 'test_results.json')

    model.to(device)
    model.eval()

    scores = []
    for _ in range(repeats):
        all_y_pred = []
        all_y_true = []
        for data in iter:
            data = data.to(device)
            y_true = data.y

            data = model(data, walk_steps=steps)

            y_pred = torch.sigmoid(data.y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            all_y_pred.append(y_pred)
            all_y_true.append(y_true)

        score = evaluator.eval({"y_true": np.vstack(all_y_true), "y_pred": np.vstack(all_y_pred)})
        scores.append(score[score_key])

    score = np.mean(scores)
    std = np.std(scores)

    print(f'Test Score: {score:.4f} (+-{std:.4f})')

    with open(res_path, 'w') as f:
        json.dump({'score': float(score), 'std': float(std)}, f, indent=4)

    return score, std


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="path to the model directories")
    parser.add_argument("--data", type=str, default='MOLPCBA', choices={'MOLHIV', 'MOLPCBA'}, help="OGB Dataset to use")
    parser.add_argument("--reps", type=int, default=10, help="Number of eval repetitions")
    parser.add_argument("--steps", type=int, default=150, help="Number of steps in each walk")
    parser.add_argument("--batch_size", type=int, default=60, help="Batch size used for testing")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--split", type=str, default='test', choices={'test', 'val'}, help="split to evaluate on")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = args.model_dir
    PCKL_PATH = f'data/OGB/{args.data}.pckl'

    ogb_data_name = 'ogbg-molhiv' if args.data == 'MOLHIV' else 'ogbg-molpcba'
    score_key = score_keys[args.data]
    evaluator = Evaluator(ogb_data_name)

    print('Loading Graphs...')

    _, _, val_graphs, test_graphs = torch.load(PCKL_PATH)
    iter = CRaWlLoader(test_graphs if args.split == 'test' else val_graphs, batch_size=args.batch_size, num_workers=8)

    mean_list, std_list = [], []
    model_list = sorted(list(glob(args.model_dir)))
    for model_dir in model_list:
        print(f'Evaluating {model_dir}...')
        model = CRaWl.load(model_dir)
        model.model_dir = model_dir
        model.save()

        mean, std = test(model, iter, repeats=args.reps, steps=args.steps)

        mean_list.append(mean)
        std_list.append(std)

    print(f'Mean Score {np.mean(mean_list):.5f} (+-{np.std(mean_list):.5f} CMD) (+-{np.mean(std_list):.5f} IMD)')
