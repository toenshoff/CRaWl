import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

default_device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


def clear_data(data_dict):
    for x in data_dict.values():
        del x


def eval(model, iter, repeats=5, device=default_device):
    model.eval()

    binary = model.out_dim == 1

    accs = []
    for _ in range(repeats):
        eq = 0
        total = 0
        for data in iter:
            data = data.to(device)
            y = data.y
            data = model(data)

            if binary:
                y_pred = torch.sigmoid(data.y_pred)
                correct = torch._cast_Int(y_pred > 0.5).eq(y).sum()
            else:
                correct = data.y_pred.argmax(dim=1).eq(y.reshape(-1)).sum()

            eq += correct.cpu().detach().numpy()
            total += float(y.shape[0])

        accs.append(eq / float(total))
    return np.mean(accs), np.std(accs)


def train(model, train_iter, val_iter, device=default_device):
    writer = SummaryWriter(model.model_dir)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=model.config['lr'], weight_decay=model.config['weight_decay'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=model.config['decay_factor'], patience=model.config['patience'], verbose=True, mode='max')

    max_epochs = model.config['epochs']
    walk_start_p = model.config['train_start_ratio']

    best_val_acc = 0.0

    for e in range(1, max_epochs+1):
        train_acc = []
        train_loss = []

        model.train()
        for data in train_iter:
            data = data.to(device)
            opt.zero_grad()

            data = model(data, walk_start_p=walk_start_p)

            data.y = data.y.view(-1)
            loss = model.loss(data)
            loss.backward()
            opt.step()

            acc = data.y_pred.argmax(dim=1).eq(data.y.reshape(-1)).sum() / float(data.y.shape[0])
            train_acc.append(acc.cpu().detach().numpy())
            train_loss.append(loss.cpu().detach().numpy())

        train_acc = np.mean(train_acc)
        train_loss = np.mean(train_loss)
        val_acc, val_std = eval(model, val_iter, repeats=model.config['eval_rep'])

        writer.add_scalar('Loss/train', train_loss, e)
        writer.add_scalar('Acc/train', train_acc, e)
        writer.add_scalar('Acc/val', val_acc, e)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            model.save()

        print(f'Epoch {e + 1} Loss: {train_loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f} (+-{val_std:.4f})')

        sch.step(val_acc)
        if sch.state_dict()['_last_lr'][0] < 0.00001 or e > max_epochs:
            break

