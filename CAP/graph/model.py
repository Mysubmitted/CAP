# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool

def _mlp(d_in, d_hidden, d_out, bn=True):
    layers = [nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm1d(d_hidden))
    layers.append(nn.Linear(d_hidden, d_out))
    return nn.Sequential(*layers)

POOLERS = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}

class GIN5(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, drop_ratio: float = 0.5):
        super().__init__()
        self.drop = drop_ratio
        self.in_dim = input_dim
        self.out_dim = hidden_dim

        self.gin = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * 5
        for i in range(5):
            self.gin.append(GINConv(_mlp(dims[i], hidden_dim, hidden_dim, bn=True), train_eps=True))
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(5)])

    def forward(self, data, prompt_type=None, prompt=None, pooling=False, pool="mean"):
        assert pooling in ['mean', 'sum', 'max', False]
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if prompt is not None and prompt_type in ['SGP', 'GPF']:
            prompt.compute_S(x, edge_index)
            x = prompt(x, layer=0)  

        x = self.gin[0](x, edge_index); x = self.bns[0](x); x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.drop, training=self.training)

        if prompt is not None and prompt_type in ['SGP', 'GPF']:
            x = prompt(x, layer=1)

        for i in range(1, 5):
            x = self.gin[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            if i < 4:
                x = F.dropout(x, p=self.drop, training=self.training)

        if not pooling:
            return x
        g = POOLERS[pool](x, batch)
        return g

class GraphClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, drop_ratio: float = 0.5, pool: str = "mean"):
        super().__init__()
        self.gnn = GIN5(input_dim, hidden_dim, drop_ratio)
        self.pool = pool
        self.cls = nn.Linear(self.gnn.out_dim, num_classes)

    def forward(self, data, prompt_type=None, prompt=None):
        g = self.gnn(data, prompt_type=prompt_type, prompt=prompt, pooling='mean', pool=self.pool)
        logit = self.cls(g)
        return logit
