import torch
import numpy as np
import torch_geometric as pygeo
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter_sum


def preproc(data):
    """ Preprocess Pytorch Geometric data objects to be used with our walk generator """

    if not data.is_coalesced():
        data.coalesce()

    if data.num_node_features == 0:
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float32)

    if data.num_edge_features == 0:
        data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.float32)

    edge_idx = data.edge_index
    edge_feat = data.edge_attr
    node_feat = data.x

    # remove isolated nodes
    #if data.contains_isolated_nodes():
    #    edge_idx, edge_feat, mask = pygeo.utils.remove_isolated_nodes(edge_idx, edge_feat, data.num_nodes)
    #    node_feat = node_feat[mask]

    # Enforce undirected graphs
    if edge_idx.shape[1] > 0 and not pygeo.utils.is_undirected(edge_idx):
        x = edge_feat.detach().numpy()
        e = edge_idx.detach().numpy()
        x_map = {(e[0,i], e[1,i]): x[i] for i in range(e.shape[1])}
        edge_idx = pygeo.utils.to_undirected(edge_idx)
        e = edge_idx.detach().numpy()
        x = [x_map[(e[0,i], e[1,i])] if (e[0,i], e[1,i]) in x_map.keys() else x_map[(e[1,i], e[0,i])] for i in range(e.shape[1])]
        edge_feat = torch.tensor(x)

    data.edge_index = edge_idx
    data.edge_attr = edge_feat
    data.x = node_feat

    order = node_feat.shape[0]

    # create bitwise encoding of adjacency matrix using 64-bit integers
    data.node_id = torch.arange(0, order)
    bit_id = torch.zeros((order, order // 63 + 1), dtype=torch.int64)
    bit_id[data.node_id, data.node_id // 63] = torch.tensor(1) << data.node_id % 63
    data.adj_bits = scatter_sum(bit_id[edge_idx[0]], edge_idx[1], dim=0, dim_size=data.num_nodes)

    # compute node offsets in the adjacency list
    data.degrees = pygeo.utils.degree(edge_idx[0], dtype=torch.int64, num_nodes=data.num_nodes)
    adj_offset = torch.zeros((order,), dtype=torch.int64)
    adj_offset[1:] = torch.cumsum(data.degrees, dim=0)[:-1]
    data.adj_offset = adj_offset

    if not torch.is_tensor(data.y):
        data.y = torch.tensor(data.y)
    data.y = data.y.view(1, -1)

    return data


def merge_batch(graph_data):
    """ Custom function to collate preprocessed data objects in the data loader """

    adj_offset = [d.adj_offset for d in graph_data]
    degrees = [d.degrees for d in graph_data]
    edge_idx = [d.edge_index for d in graph_data]

    num_nodes = torch.tensor([d.shape[0] for d in degrees])
    num_edges = torch.tensor([e.shape[1] for e in edge_idx])
    num_graphs = len(graph_data)

    x_node = torch.cat([d.x for d in graph_data], dim=0)
    x_edge = torch.cat([d.edge_attr for d in graph_data], dim=0)
    x_edge = x_edge.view(x_edge.shape[0], -1)

    adj_offset = torch.cat(adj_offset)
    degrees = torch.cat(degrees)
    edge_idx = torch.cat(edge_idx, dim=1)

    node_graph_idx = torch.cat([i * torch.ones(x, dtype=torch.int64) for i, x in enumerate(num_nodes)])
    edge_graph_idx = torch.cat([i * torch.ones(x, dtype=torch.int64) for i, x in enumerate(num_edges)])

    node_shift = torch.zeros((len(graph_data),), dtype=torch.int64)
    edge_shift = torch.zeros((len(graph_data),), dtype=torch.int64)
    node_shift[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
    edge_shift[1:] = torch.cumsum(num_edges, dim=0)[:-1]

    adj_offset += edge_shift[node_graph_idx]
    edge_idx += node_shift[edge_graph_idx].view(1, -1)

    graph_offset = node_shift

    adj_bits = [d.adj_bits for d in graph_data]
    max_enc_length = np.max([p.shape[1] for p in adj_bits])
    adj_bits = torch.cat([F.pad(b, (0,max_enc_length-b.shape[1],0,0), 'constant', 0) for b in adj_bits], dim=0)

    node_id = torch.cat([d.node_id for d in graph_data], dim=0)

    y = torch.cat([d.y for d in graph_data], dim=0)

    data = Data(x=x_node, edge_index=edge_idx, edge_attr=x_edge, y=y)
    data.batch = node_graph_idx
    data.edge_batch = edge_graph_idx
    data.adj_offset = adj_offset
    data.degrees = degrees
    data.graph_offset = graph_offset
    data.order = num_nodes
    data.num_graphs = num_graphs
    data.node_id = node_id
    data.adj_bits = adj_bits
    return data


class CRaWlLoader(DataLoader):
    """ Custom Loader for our data objects """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(CRaWlLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=merge_batch, **kwargs)