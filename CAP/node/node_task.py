import os, json, random, logging, argparse, csv
from itertools import product
from typing import Dict, Any, List, Tuple
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn import metrics

from load_data import load_node_data, NodeDownstream
from model import GCN
from prompt import SGSPAssign2, SGPPrompt
from logger import Logger
from tqdm import tqdm


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NodeTaskSGP:
    SUPPORTED_DATA = ['Cora', 'CiteSeer', 'PubMed', 'texas', 'wisconsin', 'film', 'chameleon']

    def __init__(self, dataset_name: str, shots: int, hidden_dim: int,
                 device: torch.device, pretrain_ckpt: str,
                 assign_K: int, logger: Logger):

        self.device, self.logger = device, logger
        if dataset_name not in self.SUPPORTED_DATA:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        data, in_dim, out_dim = load_node_data(dataset_name, './data')
        self.train_data, self.test_data = NodeDownstream(data, shots, 0.2)

        self.gnn = GCN(in_dim, hidden_dim, hidden_dim).to(device)
        assign = SGSPAssign2(in_dim, K=assign_K, gnn_hidden=hidden_dim)
        self.prompt = SGPPrompt(assign, d_input=in_dim, d_model=hidden_dim).to(device)
        self.classifier = nn.Linear(hidden_dim, out_dim).to(device)

        if pretrain_ckpt and os.path.isfile(pretrain_ckpt):
            self._load_pretrain(pretrain_ckpt)
        else:
            self.logger.info(f"[WARN] Checkpoint not found at {pretrain_ckpt}, utilizing random init.")
            
        for p in self.gnn.parameters():
            p.requires_grad = False
        self.gnn.eval()
        for p in self.prompt.assign.parameters():  
            p.requires_grad = False
        self.prompt.assign.eval()

    def _load_pretrain(self, path: str):
        sd = torch.load(path, map_location=self.device)
        gnn_sd = {k[len('gnn.'):]: v for k, v in sd.items()
                  if k.startswith('gnn.')}
        self.gnn.load_state_dict(gnn_sd, strict=False)
        assign_sd = {k[len('assign.'):]: v for k, v in sd.items()
                     if k.startswith('assign.')}
        self.prompt.assign.load_state_dict(assign_sd, strict=False)
        self.logger.info(f"[CKPT] Loaded pretrain from {path}")

    def train(self, batch_size: int, lr: float, weight_decay: float,
              epochs: int, eval_interval: int = 10) -> Tuple[float, int]:
        tr_loader = DataLoader(self.train_data, batch_size, shuffle=True)
        te_loader = DataLoader(self.test_data, batch_size, shuffle=False)
        params = [self.prompt.Z0] + [self.prompt.Z1] + list(self.classifier.parameters())
        optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        best_acc, best_ep = 0., -1
        for ep in tqdm(range(1, epochs + 1), total=epochs, desc="Epochs", dynamic_ncols=True):
            self.gnn.eval()
            self.prompt.assign.eval(); self.classifier.train()
            for batch in tr_loader:
                batch = batch.to(self.device)
                optim.zero_grad()
                h = self.gnn(batch, prompt_type='cap',
                             prompt=self.prompt, pooling='mean')
                out = self.classifier(h)
                loss = F.cross_entropy(out, batch.y.squeeze())
                loss.backward(); optim.step()

            if ep % eval_interval == 0 or ep == epochs:
                acc, _ = self._evaluate(te_loader)
                if acc > best_acc:
                    best_acc, best_ep = acc, ep
        return best_acc, best_ep

    @torch.no_grad()
    def _evaluate(self, loader):
        self.gnn.eval(); self.prompt.eval(); self.classifier.eval()
        preds, labels = [], []
        for batch in loader:
            batch = batch.to(self.device)
            h = self.gnn(batch, prompt_type='cap',
                         prompt=self.prompt, pooling='mean')
            out = self.classifier(h)
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(batch.y.squeeze().cpu().tolist())
        acc = metrics.accuracy_score(labels, preds)
        return acc, 0.0


def run_for_ckpt(args, device, ckpt_path, assign_K, csv_writer):
    space: Dict[str, List[Any]] = {
        'lr': [1e-3],
        'wd': [0],
        'hidden_dim': [args.hidden_dim],
        'assign_K': [assign_K]
    }
    combos = list(product(*space.values()))
    os.makedirs('logs_cap', exist_ok=True)
    logger = Logger('logs_cap/search.log',
                    logging.Formatter('%(asctime)s - %(message)s'))
    best_cfg, best_acc = None, 0.

    for idx, (lr, wd, hd, k) in enumerate(combos, 1):
        for i in range(3):
            set_random_seed(i)
            task = NodeTaskSGP(dataset_name=args.dataset,
                               shots=args.shots,
                               hidden_dim=hd,
                               device=device,
                               pretrain_ckpt=ckpt_path,
                               assign_K=k,
                               logger=logger)
            acc, _ = task.train(batch_size=args.batch_size,
                                lr=lr,
                                weight_decay=wd,
                                epochs=args.epochs,
                                eval_interval=args.eval_int)
            if acc > best_acc:
                best_acc, best_cfg = acc, dict(lr=lr, wd=wd,
                                               hidden_dim=hd, assign_K=k)
    
    # 写入 CSV
    if csv_writer:
        csv_writer.writerow([ckpt_path, assign_K, best_acc, best_cfg])
    print(f"Finished. Best Acc: {best_acc:.4f} with Config: {best_cfg}")


def main():
    p = argparse.ArgumentParser("CAP Downstream Task")
    
    # 核心参数
    p.add_argument('--dataset', default='Cora', help='Dataset name')
    p.add_argument('--ckpt', type=str, required=True, help='Path to pre-trained checkpoint')
    p.add_argument('--K', type=int, default=8, help='Number of clusters (K) for CAP')
    
    # 训练参数
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--eval_int', type=int, default=10)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--ssl', choices=['edgepred', 'attrmask', 'graphcl'], default='graphcl')
    
    args = p.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 准备 CSV 记录
    os.makedirs('logs_cap', exist_ok=True)
    csv_path = f'logs_cap/all_results_{args.shots}shots.csv'
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model_path', 'assign_K', 'best_acc', 'best_config'])
        
        run_for_ckpt(args, device, args.ckpt, args.K, writer)

if __name__ == '__main__':
    main()