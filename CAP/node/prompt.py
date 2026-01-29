import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops
from torch_geometric.data   import Data

class EdgePrompt(nn.Module):
    def __init__(self, dim_list):
        super().__init__()
        self.global_prompt = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, dim)) for dim in dim_list]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for prompt in self.global_prompt:
            glorot(prompt)

    def get_prompt(self, x, edge_index, layer):
        return self.global_prompt[layer]


class EdgePromptplus(nn.Module):
    def __init__(self, dim_list, num_anchors):
        super().__init__()
        self.anchor_prompt = nn.ParameterList(
            [nn.Parameter(torch.Tensor(num_anchors, dim)) for dim in dim_list]
        )
        self.w = nn.ModuleList([nn.Linear(2 * dim, num_anchors) for dim in dim_list])
        self.reset_parameters()

    def reset_parameters(self):
        for anchor in self.anchor_prompt:
            glorot(anchor)
        for w in self.w:
            w.reset_parameters()

    def get_prompt(self, x, edge_index, layer):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        combined_x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        b = F.softmax(F.leaky_relu(self.w[layer](combined_x)), dim=1)
        prompt = b @ self.anchor_prompt[layer]
        return prompt

class SGSPAssign2(nn.Module):
    def __init__(self,
                 in_dim: int,
                 K: int,
                 gnn_hidden: int = 128,
                 learn_tau: bool = True,
                 init_tau: float = 1.0):
        super().__init__()
        from model import GCN                   
        self.gnn_a = GCN(in_dim, gnn_hidden, gnn_hidden)
        self.K = K
        self.W = nn.Linear(gnn_hidden, K, bias=False)

        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau))) if learn_tau else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.gnn_a(Data(x=x, edge_index=edge_index), pooling=False)   # [N, d]
        logits = self.W(h)                                               # Z: [N, K]
        tau = self.log_tau.exp() if self.log_tau is not None else 1.0
        S = F.softmax(logits / tau, dim=-1)                              # [N, K]
        return S, logits

class SGPPrompt(nn.Module):
    def __init__(
        self,
        assign: SGSPAssign2,
        d_input: int,
        d_model: int,
        init_std: float = 2e-2
    ):
        super().__init__()
        self.assign = assign
        K = assign.K
        # learnable prompt tokens
        self.Z0 = nn.Parameter(torch.randn(K, d_input) * init_std)
        self.Z1 = nn.Parameter(torch.randn(K, d_model) * init_std)
        self._S = None  # cache S for inspection
    def compute_S(self, X: torch.Tensor, edge_index: torch.Tensor):
        S, _ = self.assign(X, edge_index)
        self._S = S
        return S

    def forward(self, X: torch.Tensor, layer: int = 0):
        if layer == 0:
            return X + self._S @ self.Z0              # ΔX = S·Z0
        return X + self._S @ self.Z1              # ΔX = S·Z1


class LinkReconLoss(nn.Module):
    def forward(self, data: Data, S: torch.Tensor):
        N = S.size(0)
        A = torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones(data.edge_index.size(1), device=S.device),
            (N, N)
        ).to_dense()
        A = A + torch.eye(N, device=S.device)           # Â = A + I
        A = A / A.sum(-1, keepdim=True).clamp(min=1)    # row-norm
        recon = S @ S.t()
        return (recon - A).pow(2).mean()


class EntropyLoss(nn.Module):
    def forward(self, S: torch.Tensor, eps: float = 1e-10):
        p = S.clamp(min=eps)
        return -(p * p.log()).sum(dim=1).mean()
