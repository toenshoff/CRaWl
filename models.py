import os
import torch
from torch_scatter import scatter_mean, scatter_sum
from walker import Walker


class VNUpdate(torch.nn.Module):
    def __init__(self, dim, config):
        """
        Intermediate update layer for the virtual node
        :param dim: Dimension of the latent node embeddings
        :param config: Python Dict with the configuration of the CRaWl network
        """
        super(VNUpdate, self).__init__()

        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim, bias=False),
                                       torch.nn.BatchNorm1d(dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(config['dropout']),
                                       torch.nn.Linear(dim, dim, bias=False))

    def forward(self, data):
        x = scatter_sum(data.h, data.batch, dim=0)
        if data.vn_h is not None:
            x = x + data.vn_h
        data.vn_h = self.mlp(x)
        data.h = data.h + data.vn_h[data.batch]
        return data


class ConvModule(torch.nn.Module):
    def __init__(self, conv_dim, w_feat_dim, dim_in, dim_out, kernel_size, config):
        """
        :param conv_dim: Hidden dimension of the convolutions
        :param w_feat_dim: feature dimension of the walk feature tensors (without the node features)
        :param dim_in: Dimension of the latent node embedding used as input
        :param dim_out: Dimension of updated latent node embedding
        :param kernel_size: Kernel size of the convolutions
        :param config: Python Dict with the configuration of the CRaWl network
        """
        super(ConvModule, self).__init__()

        self.rescale = not dim_out == dim_in
        if self.rescale:
            self.rescale_op = torch.nn.Linear(dim_in, dim_out, bias=False)

        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(dim_in + w_feat_dim, conv_dim, kernel_size, padding=0, bias=False),
            torch.nn.BatchNorm1d(conv_dim),
            torch.nn.ReLU(),
            torch.nn.Conv1d(conv_dim, conv_dim, kernel_size, padding=0, bias=False),
            torch.nn.BatchNorm1d(conv_dim),
            torch.nn.ReLU())

        self.out = torch.nn.Sequential(torch.nn.Linear(conv_dim, 2 * dim_out, bias=False),
                                       torch.nn.BatchNorm1d(2 * dim_out),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(config['dropout']),
                                       torch.nn.Linear(2 * dim_out, dim_out, bias=False))

    def forward(self, data):
        # rescale for the residual connection
        if self.rescale:
            h_r = self.rescale_op(data.h)
        else:
            h_r = data.h

        # build walk feature tensor
        h_c = torch.cat([data.h[data.walk_nodes].transpose(2, 1), data.walk_x], dim=1)

        # apply the cnn
        h_c = self.convs(h_c)

        # pool in walklet embeddings into nodes
        wl = h_c.shape[0] * h_c.shape[2]
        h = scatter_mean(h_c.transpose(2, 1).reshape(wl, -1), data.walk_nodes_flatt, dim=0, dim_size=data.num_nodes)

        # update embeddings
        h = self.out(h)
        data.h = h + h_r
        return data


class CRaWl(torch.nn.Module):
    def __init__(self, model_dir, config, node_feat_dim, edge_feat_dim, out_dim, node_feat_enc=None, edge_feat_enc=None):
        """
        :param model_dir: Directory to store model in
        :param config: Python Dict that specifies the configuration of the model
        :param node_feat_dim: Dimension of the node features
        :param edge_feat_dim: Dimension of the edge features
        :param out_dim: Output dimension
        :param node_feat_enc: Optional initial embedding of node features
        :param edge_feat_enc: Optional initial embedding of edge features
        """
        super(CRaWl, self).__init__()
        self.model_dir = model_dir
        self.config = config
        self.out_dim = out_dim
        self.node_feat_enc = node_feat_enc
        self.edge_feat_enc = edge_feat_enc
        self.layers = config['layers']
        self.hidden = config['hidden_dim']
        self.kernel_size = config['kernel_size']
        self.dropout = config['dropout']
        self.residual = config['residual'] if 'residual' in config.keys() else True
        self.pool_op = config['pool'] if 'pool' in config.keys() else 'mean'
        self.vn = config['vn'] if 'vn' in config.keys() else False

        self.border = 2 * ((self.kernel_size - 1) // 2)
        self.tail = (self.kernel_size - 1) * 2

        self.walker = Walker(config)

        self.walk_dim = self.walker.struc_feat_dim + edge_feat_dim
        self.conv_dim = config['conv_dim'] if 'conv_dim' in config.keys() else self.hidden

        modules = []
        for i in range(self.layers):
            modules.append(ConvModule(self.conv_dim, self.walk_dim, node_feat_dim if i == 0 else self.hidden, self.hidden, self.kernel_size, config))
            if self.vn and i < self.layers - 1:
                modules.append(VNUpdate(self.hidden, config))

        self.convs = torch.nn.Sequential(*modules)

        self.node_out = torch.nn.Sequential(torch.nn.BatchNorm1d(self.hidden), torch.nn.ReLU())

        if config['graph_out'] == 'linear':
            self.linear_out = True
            self.graph_out = torch.nn.Sequential(torch.nn.Linear(self.hidden, out_dim))
        else:
            self.linear_out = False
            self.graph_out = torch.nn.Sequential(torch.nn.Linear(self.hidden, self.hidden),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(self.hidden, out_dim))

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of paramters: {pytorch_total_params}')

    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self, os.path.join(self.model_dir, 'model.pckl'))

    @staticmethod
    def load(path):
        return torch.load(os.path.join(path, 'model.pckl'))

    def forward(self, data, walk_steps=None, walk_start_p=1.0):
        # apply initial node feature encoding (optional)
        x_n = data.x
        if self.node_feat_enc is not None:
            x_n = self.node_feat_enc(x_n)
        data.h = x_n

        # apply initial edge feature encoding (optional)
        x_e = data.edge_attr
        if self.edge_feat_enc is not None:
            x_e = self.edge_feat_enc(x_e)

        # compute walks
        data = self.walker.sample_walks(data, x_e, steps=walk_steps, start_p=walk_start_p)

        # pre-compute array of center nodes across all walks
        data.walk_nodes_flatt = data.walk_nodes[:, self.border:-self.border].reshape(-1)

        if self.vn:
            data.vn_h = None

        # apply convolutions
        self.convs(data)

        # pool node embeddings
        data.h = self.node_out(data.h)
        if self.pool_op == 'sum':
            x = scatter_sum(data.h, data.batch, dim=0)
        else:
            x = scatter_mean(data.h, data.batch, dim=0)

        y = self.graph_out(x)
        return y
