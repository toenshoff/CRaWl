import torch
import numpy as np
import os
import json

default_device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, iter, repeats=5, steps=50, device=default_device):
    res_path = os.path.join(model.model_dir, 'test_results.json')

    model.eval()
    accs = []
    for _ in range(repeats):
        eq = 0
        total = 0
        for data in iter:
            data = data.to(device)
            y = data.y
            data = model(data, steps)

            correct = data.y_pred.argmax(dim=1).eq(y.view(-1)).sum()

            eq += correct.cpu().detach().numpy()
            total += float(y.shape[0])

        accs.append(eq / float(total))

    acc = np.mean(accs)
    std = np.std(accs)

    print(f'Test Score: {acc:.4f} (+-{std:.4f})')

    with open(res_path, 'w') as f:
        json.dump({'score': float(acc), 'std': float(std)}, f, indent=4)

    return acc, std
