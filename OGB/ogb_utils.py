from ogb.graphproppred import PygGraphPropPredDataset
from data_utils import preproc
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
import torch
from tqdm import tqdm

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

hiv_atom_labels = [[0,2,4,5,6,7,8,10,11,12,13,14,15,16,18,19,21,23,24,25,26,27,28,29,30,31,32,33,34,39,41,43,44,45,46,49,50,52,54,64,66,73,74,76,77,78,79,80,81,82,88],
                   [0],
                   [0,1,2,3,4,5,6,7,8,9],
                   [0,1,2,3,4,5,6,7,8],
                   [0,1,2,3],
                   [0,2],
                   [0,1,2,3,4,5],
                   [0,1],
                   [0,1]]

pcba_atom_labels= [[0,4,5,6,7,8,10,12,13,14,15,16,21,23,24,25,26,27,28,29,30,31,32,33,34,37,43,46,47,48,49,50,52,55,59,63,73,77,78,79,82],
                   [0,1,2],
                   [0,1,2,3,4,5,6],
                   [4,5,6,7,8,9,11],
                   [0,1,2,3,4],
                   [0,1,2,3,4],
                   [0,1,2,3,4,5],
                   [0,1],
                   [0,1]]

hiv_bond_labels = [[0,1,2,3],
                   [0],
                   [0,1]]

pcba_bond_labels= [[0,1,2,3],
                   [0,1,2],
                   [0,1]]

class OneHotEncoder(torch.nn.Module):

    def __init__(self, train_labels, num_poss_labels, device):
        super(OneHotEncoder, self).__init__()
        self.train_labels = train_labels
        self.used_dims = [i for i, l in enumerate(train_labels) if len(l) > 1]

        self.one_hot = [torch.zeros((num_poss_labels[i], len(train_labels[i])), dtype=torch.float32, device=device) for i in range(len(train_labels))]
        for i in self.used_dims:
            self.one_hot[i][torch.tensor(train_labels[i]), torch.arange(len(train_labels[i]))] = 1.0

    def dim(self):
        return sum([len(self.train_labels[i]) for i in self.used_dims])

    def forward(self, x):
        return torch.cat([self.one_hot[i][x[:, i]] for i in self.used_dims], dim=1)


def get_hiv_encoders(device):
    return OneHotEncoder(hiv_atom_labels, full_atom_feature_dims, device), OneHotEncoder(hiv_bond_labels, full_bond_feature_dims, device)


def get_pcba_encoders(device):
    return OneHotEncoder(pcba_atom_labels, full_atom_feature_dims, device), OneHotEncoder(pcba_bond_labels, full_bond_feature_dims, device)


def load_graphs(ogb_name):

    dataset = PygGraphPropPredDataset(ogb_name, root='data', transform=preproc)
    out_dim = dataset[0].y.shape[1]

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    print("Preprocessing Graphs...")
    train_graphs = list(tqdm(dataset[train_idx]))
    train_graphs = [d for d in train_graphs if d.num_edges > 0]
    valid_graphs = list(dataset[valid_idx])
    test_graphs = list(dataset[test_idx])

    return out_dim, train_graphs, valid_graphs, test_graphs

