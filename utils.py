import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from math import ceil

def load_data(ds_name, use_node_labels):
    node2graph = {}
    Gs = []
    
    with open("datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name), "r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1
    
    with open("datasets/%s/%s_A.txt"%(ds_name,ds_name), "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])]-1].add_edge(int(edge[0]), int(edge[1]))
    
    if use_node_labels:
        with open("datasets/%s/%s_node_labels.txt"%(ds_name,ds_name), "r") as f:
            c = 1
            for line in f:
                node_label = int(line[:-1])
                Gs[node2graph[c]-1].node[c]['label'] = node_label
                c += 1
        
    labels = []
    with open("datasets/%s/%s_graph_labels.txt"%(ds_name,ds_name), "r") as f:
        for line in f:
            labels.append(int(line[:-1]))
    
    labels  = np.array(labels, dtype = np.float)
    return Gs, labels


def process_node_labels(Gs):
    node_labels = dict()
    for G in Gs:
        for node in G.nodes():
            if G.node[node]["label"] not in node_labels:
                node_labels[G.node[node]["label"]] = len(node_labels)

    n_node_labels = len(node_labels)
    for G in Gs:
        for node in G.nodes():
            G.node[node]["label"] = node_labels[G.node[node]["label"]]

    return Gs, n_node_labels


def generate_batches(Gs, use_node_labels, n_feat, y, batch_size, radius, device, shuffle=False):
    N = len(Gs)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    n_batches = ceil(N/batch_size)

    adj_lst = list()
    features_lst = list()
    idx_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_nodes = 0
        for j in range(i, min(i+batch_size, N)):
            n_nodes += Gs[index[j]].number_of_nodes()

        y_batch = np.zeros(min(i+batch_size, N)-i)
        idx_batch = np.zeros(min(i+batch_size, N)-i+1, dtype=np.int64)
        idx_batch[0] = 0
        idx_node = np.zeros(n_nodes, dtype=np.int64)

        edges_batch = list()
        for _ in range(radius*2):
            edges_batch.append(list())
        
        tuple_to_idx = list()
        features_batch = list()
        for _ in range(radius+1):
            tuple_to_idx.append(dict())
            features_batch.append(list())
        
        for j in range(i, min(i+batch_size, N)):
            n = Gs[index[j]].number_of_nodes()
            feat = dict()
            if use_node_labels:
                for node in Gs[index[j]].nodes():
                    v = np.zeros(n_feat)
                    v[Gs[index[j]].node[node]["label"]] = 1
                    feat[node] = v
            else:
                for node in Gs[index[j]].nodes():
                    feat[node] = [Gs[index[j]].degree(node)]

            for k,n1 in enumerate(Gs[index[j]].nodes()):
                idx_node[idx_batch[j-i]+k] = j-i
                
                ego = nx.ego_graph(Gs[index[j]], n1, radius=radius)
                dists = nx.single_source_shortest_path_length(ego, n1)

                for n2 in ego.nodes():
                    tuple_to_idx[dists[n2]][(n1,n2)] = len(tuple_to_idx[dists[n2]])
                    features_batch[dists[n2]].append(feat[n2])

                for n2 in ego.nodes():
                    for n3 in ego.neighbors(n2):
                        if dists[n3] > dists[n2]:
                            edges_batch[2*dists[n2]].append((tuple_to_idx[dists[n2]][(n1,n2)], tuple_to_idx[dists[n2]+1][(n1,n3)]))
                        elif dists[n3] == dists[n2]:
                            edges_batch[2*dists[n2]-1].append((tuple_to_idx[dists[n2]][(n1,n2)], tuple_to_idx[dists[n2]][(n1,n3)]))

            idx_batch[j-i+1] = idx_batch[j-i] + n
            y_batch[j-i] = y[index[j]]

        adj_batch = list()
        for i in range(2*radius):
            if i%2 == 0:
                edges = np.vstack(edges_batch[i])
                adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(len(features_batch[i//2]), len(features_batch[i//2+1])), dtype=np.float32)
            else:
                if len(edges_batch[i]) == 0:
                    adj = sp.coo_matrix((len(features_batch[i//2+1]), len(features_batch[i//2+1])), dtype=np.float32)
                else:
                    edges = np.vstack(edges_batch[i])
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(len(features_batch[i//2+1]), len(features_batch[i//2+1])), dtype=np.float32)    
            
            adj_batch.append(sparse_mx_to_torch_sparse_tensor(adj).to(device))

        for i in range(radius+1):
            features_batch[i] = torch.FloatTensor(features_batch[i]).to(device)
        
        adj_lst.append(adj_batch)
        features_lst.append(features_batch)
        idx_lst.append(torch.LongTensor(idx_node).to(device))
        y_lst.append(torch.LongTensor(y_batch).to(device))

    return adj_lst, features_lst, idx_lst, y_lst
    

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
