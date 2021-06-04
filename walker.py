import torch
import numpy as np

MAXINT = np.iinfo(np.int64).max


class Walker(torch.nn.Module):

    def __init__(self, config):
        super(Walker, self).__init__()

        self.steps = config['steps']
        self.train_start_ratio = config['train_start_ratio']
        self.win_size = config['win_size']

        self.compute_id = 'compute_id_feat' not in config.keys() or config['compute_id_feat']
        self.compute_adj = 'compute_adj_feat' not in config.keys() or config['compute_adj_feat']

        self.struc_feat_dim = 0
        if self.compute_id:
            self.struc_feat_dim += self.win_size
        if self.compute_adj:
            self.struc_feat_dim += self.win_size - 1

        self.non_backtracking = False if config['walk_mode'] == 'uniform' else True
        self.delta = config['walk_delta'] if 'walk_delta' in config.keys() else 0.0

    @staticmethod
    def sample_start(start_p, graph_idx, graph_offset, order, device):
        """
        Randomly sample start nodes
        :param start_p: Probability of starting a walk at a node
        :param graph_idx: Assignment of nodes to graphs
        :param graph_offset: Node list offset of each graph in the batch
        :param order: Nuber of nodes in each graph of the batch
        :param device: device to construct tensors on
        :return: A tensor of start vertices (index list) and an assignment to the graphs in the batch
        """

        num_graphs = order.shape[0]
        num_nodes = graph_idx.shape[0]
        num_walks = int(np.ceil(start_p * num_nodes))
        num_extra = num_walks - num_graphs

        idx = graph_offset + (torch.randint(0, MAXINT, (num_graphs,), device=device) % order)
        idx = torch.cat([idx, torch.randperm(num_nodes, device=device)[:num_extra]])

        choices = torch.randint(0, MAXINT, (num_walks,), device=device)
        start_graph = graph_idx[idx]
        start = graph_offset[start_graph] + (choices % order[start_graph])

        del idx, choices
        return start, start_graph

    def unweighted_choice(self, i, walks, adj_nodes, adj_offset, degrees, nb_degrees, choices):
        """
        :param i: Index of the current step
        :param walks: Tensor of vertices in the walk
        :param adj_nodes: Adjacency List
        :param adj_offset: Node offset in the adjacency list
        :param choices: Cache of random integers
        :param degrees: Degree of each node
        :param nb_degrees: Reduced degrees for no-backtrack walks
        :return: A list of a chosen outgoing edge for each walk
        """
        # do uniform step
        cur_nodes = walks[i]
        edge_idx = choices[i] % degrees[cur_nodes]
        chosen_edges = adj_offset[cur_nodes] + edge_idx

        if self.non_backtracking and i > 0:
            old_nodes = walks[i - 1]
            new_nodes = adj_nodes[chosen_edges]
            # correct backtracking
            bt = new_nodes == old_nodes
            if bt.max():
                bt_nodes = walks[i][bt]
                chosen_edges[bt] = adj_offset[bt_nodes] + (edge_idx[bt] + 1 + (choices[i][bt] % nb_degrees[bt_nodes])) % degrees[bt_nodes]

        return chosen_edges

    def sample_walks(self, data, steps=None, start_p=1.0):
        """
        :param data: Preprocessed PyTorch Geometric data object.
        :param x_edge: Edge features
        :param steps: Number of walk steps (if None, default_old from config is used)
        :param start_p: Probability of starting a walk at each node
        :return: The data object with the walk added as an attribute
        """

        device = data.x.device

        # get adjacency data
        adj_nodes = data.edge_index[1]
        adj_offset = data.adj_offset
        degrees = data.degrees
        node_id = data.node_id
        adj_bits = data.adj_bits
        graph_idx = data.batch
        graph_offset = data.graph_offset
        order = data.order

        # use default_old number of steps if not specified
        if steps is None:
            steps = self.steps

        # set dimensions
        s = self.win_size
        n = degrees.shape[0]
        l = steps + 1

        # sample starting nodes
        if self.training and start_p < 1.0:
            start, walk_graph_idx = Walker.sample_start(start_p, graph_idx, graph_offset, order, device)
        else:
            start = torch.arange(0, n, dtype=torch.int64).view(-1)
        start = start[degrees[start] > 0]

        # init tensor to hold walk indices
        w = start.shape[0]
        walks = torch.zeros((l, w), dtype=torch.int64, device=device)
        walks[0] = start

        walk_edges = torch.zeros((l-1, w), dtype=torch.int64, device=device)

        # get all random decisions at once (faster then individual calls)
        choices = torch.randint(0, MAXINT, (steps, w), device=device)

        if self.compute_id:
            id_enc = torch.zeros((l, s, w), dtype=torch.bool, device=device)

        if self.compute_adj:
            edges = torch.zeros((l, s, w), dtype=torch.bool, device=device)

        # remove one choice of each node with deg > 1 for no_backtrack walks
        nb_degree_mask = (degrees == 1)
        nb_degrees = nb_degree_mask * degrees + (~nb_degree_mask) * (degrees - 1)

        for i in range(steps):
            chosen_edges = self.unweighted_choice(i, walks, adj_nodes, adj_offset, degrees, nb_degrees, choices)

            # update nodes
            walks[i+1] = adj_nodes[chosen_edges]

            # update edge features
            walk_edges[i] = chosen_edges

            o = min(s, i+1)
            prev = walks[i+1-o:i+1]

            if self.compute_id:
                # get local identity relation
                id_enc[i+1, s-o:] = torch.eq(walks[i+1].view(1, w), prev)

            if self.compute_adj:
                # look up edges in the bit-wise adjacency encoding
                cur_id = node_id[walks[i+1]]
                cur_int = (cur_id // 63).view(1, -1, 1).repeat(o, 1, 1)
                edges[i + 1, s - o:] = (torch.gather(adj_bits[prev], 2, cur_int).view(o,-1) >> (cur_id % 63).view(1,-1)) % 2 == 1

        # permute walks into the correct shapes
        data.walk_nodes = walks.permute(1, 0)
        data.walk_edges = walk_edges.permute(1, 0)

        # combine id, adj and edge features
        feat = []
        if self.compute_id:
            feat.append(torch._cast_Float(id_enc.permute(2, 1, 0)))
        if self.compute_adj:
            feat.append(torch._cast_Float(edges.permute(2, 1, 0))[:, :-1, :])
        data.walk_x = torch.cat(feat, dim=1) if len(feat) > 0 else None

        return data
