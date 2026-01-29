# prompt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data

def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m

def _tiny_mlp(d_in, d_hid, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(inplace=True), nn.Linear(d_hid, d_out))

class TinyGIN(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.conv1 = GINConv(_tiny_mlp(in_dim, hidden, hidden), train_eps=True)
        self.conv2 = GINConv(_tiny_mlp(hidden, hidden, hidden), train_eps=True)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out_dim = hidden

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = self.bn1(x); x = F.relu(x, inplace=True)
        x = self.conv2(x, edge_index); x = self.bn2(x); x = F.relu(x, inplace=True)
        return x

class SGSPAssign2(nn.Module):
    r"""
    g_φ: (X,A) -> Z ∈ ℝ^{N×K},  S = softmax(Z / τ)
    """
    def __init__(self, in_dim: int, K: int, gnn_hidden: int = 128, learn_tau: bool = True, init_tau: float = 1.0):
        super().__init__()
        self.K = K
        self.gnn_a = TinyGIN(in_dim, hidden=gnn_hidden)
        self.W = nn.Linear(self.gnn_a.out_dim, K, bias=False)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau))) if learn_tau else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.gnn_a(x, edge_index)                   # [N, d]
        logits = self.W(h)                              # [N, K]
        tau = self.log_tau.exp() if self.log_tau is not None else 1.0
        S = F.softmax(logits / tau, dim=-1)             # [N, K]
        return S, logits

class SGPPrompt(nn.Module):
    def __init__(self, assign: SGSPAssign2, d_input: int, d_model: int, init_std: float = 2e-2):
        super().__init__()
        self.assign = assign
        K = assign.K
        self.Z0 = nn.Parameter(torch.randn(K, d_input) * init_std)
        self.Z1 = nn.Parameter(torch.randn(K, d_model) * init_std)
        self._S = None

    @torch.no_grad()
    def compute_S(self, X: torch.Tensor, edge_index: torch.Tensor):
        S, _ = self.assign(X, edge_index)
        self._S = S
        return S

    def forward(self, X: torch.Tensor, layer: int = 0):
        if self._S is None:
            raise RuntimeError("Call compute_S(X, edge_index) before using SGPPrompt.")
        if layer == 0:
            return X + self._S @ self.Z0
        elif layer == 1:
            return X + self._S @ self.Z1
        else:
            return X

class GPFPrompt(nn.Module):
    r"""
    layer=0: X ← X + 1·p0^T
    layer=1: X ← X + 1·p1^T
    """
    def __init__(self, d_input: int, d_model: int, init_std: float = 2e-2):
        super().__init__()
        self.p0 = nn.Parameter(torch.randn(d_input) * init_std)
        self.p1 = nn.Parameter(torch.randn(d_model) * init_std)

    @torch.no_grad()
    def compute_S(self, X: torch.Tensor, edge_index: torch.Tensor):
        return None  

    def forward(self, X: torch.Tensor, layer: int = 0):
        if layer == 0:
            return X + self.p0.unsqueeze(0).expand_as(X)
        elif layer == 1:
            return X + self.p1.unsqueeze(0).expand_as(X)
        else:
            return X

def build_prompt(prompt_type: str, *, in_dim: int, model_dim: int, assigner: nn.Module = None):
    t = (prompt_type or "").lower()
    if t in ["sgp", "structure", "cap", "sgpprompt"]:
        assert assigner is not None
        return SGPPrompt(assign=assigner, d_input=in_dim, d_model=model_dim)
    elif t in ["gpf", "global"]:
        return GPFPrompt(d_input=in_dim, d_model=model_dim)
    elif t in ["none", "", None]:
        return None
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

class LinkReconLoss(nn.Module):
    def forward(self, data: Data, S: torch.Tensor):
        N = S.size(0)
        A = torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones(data.edge_index.size(1), device=S.device),
            (N, N)
        ).to_dense()
        A = A + torch.eye(N, device=S.device)
        A = A / A.sum(-1, keepdim=True).clamp(min=1)
        recon = S @ S.t()
        return (recon - A).pow(2).mean()

class EntropyLoss(nn.Module):
    r"""L_ent = − (1/N) Σ_i Σ_k S_ik log S_ik"""
    def forward(self, S: torch.Tensor, eps: float = 1e-10):
        p = S.clamp(min=eps)
        return -(p * p.log()).sum(dim=1).mean()
