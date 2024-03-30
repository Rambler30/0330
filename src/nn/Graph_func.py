from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import csgraph
import torch

def get_neight_node(pos, edge_index):
    # 获取邻接节点
    edge = np.array(edge_index.cpu())
    edge_matrix = coo_matrix((np.ones(edge.shape[1]), 
                (edge[0], edge[1]))).toarray()
    adj = csgraph.laplacian(edge_matrix, normed=True)
    adj = torch.tensor(adj).double().cuda()
    t = True
    count = set(edge[0])
    max_count = 0
    for i in count:
        len = np.argwhere(edge[0]==i).shape[0]
        if len > max_count:
            max_count = len

    K = max_count
    if t == True:
        group_idx = torch.full((pos.shape[0],K, 1), -1) # [n,k]
        grouped_xyz = torch.zeros((pos.shape[0], K, 3))  # [N1, K, 3]
        for i in range(pos.shape[0]):
            s_idx = torch.argwhere(edge_index[0]==i)
            t_idx = edge_index[1, s_idx].squeeze(-1)
            t_idx = t_idx[t_idx!=i]
            grouped_xyz[i, 0,:] = pos[i]
            grouped_xyz[i, 1:t_idx.shape[0]+1,:] = pos[t_idx, :]
            group_idx[i, 0] = i
            group_idx[i, 1:t_idx.shape[0]+1] = t_idx
        grouped_xyz, group_idx = grouped_xyz.permute(2,0,1).unsqueeze(0), group_idx.unsqueeze(0)
        # [b,3,n,k], [b,n,k]