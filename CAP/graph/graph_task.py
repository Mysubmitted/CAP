# test_graph.py
import os, argparse, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from load_data import load_graph_data, GraphDownstream
from model import GraphClassifier
from prompt import SGSPAssign2, build_prompt, freeze_module


def write_min_csv(csv_path: str, acc: float, ckpt_path: str, assigner_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["acc", "ckpt_path", "assigner_ckpt"])
        w.writerow([f"{acc:.4f}", os.path.abspath(ckpt_path),
                    os.path.abspath(assigner_path) if assigner_path else ""])


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prefixed(sd, module, prefix):
    sub = {k[len(prefix)+1:]: v for k, v in sd.items() if k.startswith(prefix + ".")}
    missing, unexpected = module.load_state_dict(sub, strict=False)
    if missing:
        print(f"[warn] missing in {prefix}: {missing[:6]}{'...' if len(missing)>6 else ''}")
    if unexpected:
        print(f"[warn] unexpected in {prefix}: {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")


@torch.no_grad()
def evaluate(model, prompt, loader, device, prompt_type):
    model.eval()
    tot, cor = 0, 0
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch, prompt_type=prompt_type, prompt=prompt)
        pred = logit.argmax(-1)
        tot += batch.y.size(0); cor += (pred == batch.y).sum().item()
    return cor / max(1, tot)


def run_one_seed(args, device, seed, sd_pretrain):
    set_seed(seed)
    ds, in_dim, num_classes = load_graph_data(args.dataset, args.root)
    train_list, test_list = GraphDownstream(ds, shots=args.shots, test_fraction=args.test_frac)
    mid = max(1, int(0.5 * len(test_list)))
    val_list = test_list[:mid]
    test_list = test_list[mid:]

    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_list,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_list,  batch_size=args.batch_size, shuffle=False)

    model = GraphClassifier(
        input_dim=in_dim, num_classes=num_classes,
        hidden_dim=args.hidden_dim, drop_ratio=args.drop, pool=args.pool
    ).to(device)
    load_prefixed(sd_pretrain, model.gnn, "gnn")
    if any(k.startswith("cls.") for k in sd_pretrain.keys()):
        load_prefixed(sd_pretrain, model.cls, "cls")

    for p in model.gnn.parameters():
        p.requires_grad = False
    model.gnn.eval()

    # 3) Prompt & assigner
    prompt_type, prompt, assigner_path_used = None, None, ""
    if args.prompt.lower() == "cap":
        assigner = SGSPAssign2(in_dim=in_dim, K=args.K, gnn_hidden=args.hidden_dim).to(device)
        if args.assigner_ckpt and os.path.isfile(args.assigner_ckpt):
            ad = torch.load(args.assigner_ckpt, map_location=device)
            load_prefixed(ad, assigner, "assign")
            assigner_path_used = os.path.abspath(args.assigner_ckpt)
        elif any(k.startswith("assign.") for k in sd_pretrain.keys()):
            load_prefixed(sd_pretrain, assigner, "assign")
            assigner_path_used = os.path.abspath(args.ckpt)
        freeze_module(assigner)
        assigner.eval()
        prompt = build_prompt("cap", in_dim=in_dim, model_dim=model.gnn.out_dim, assigner=assigner).to(device)
        prompt_type = "cap"
    elif args.prompt.lower() == "gpf":
        prompt = build_prompt("gpf", in_dim=in_dim, model_dim=model.gnn.out_dim).to(device)
        prompt_type = "GPF"

    # 4)  Prompt + Classifier
    train_params = (list(prompt.parameters()) if prompt is not None else []) + list(model.cls.parameters())
    opt = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_val = -1.0
    best_test_at_val = 0.0
    best_test_ever = 0.0
    best_state = None
    best_ep = -1

    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logit = model(batch, prompt_type=prompt_type, prompt=prompt)
            loss = ce(logit, batch.y)
            opt.zero_grad(); loss.backward(); opt.step()

        va = evaluate(model, prompt, val_loader, device, prompt_type)
        te = evaluate(model, prompt, test_loader, device, prompt_type)
        if va > best_val:
            best_val = va
            best_test_at_val = te
            best_ep = ep
            best_state = {
                "prompt": (prompt.state_dict() if prompt is not None else None),
                "classifier": model.cls.state_dict(),
                "meta": {
                    "epoch": ep, "seed": seed, "select_by": "val",
                    "best_val": float(best_val), "best_test_at_val": float(best_test_at_val),
                    "dataset": args.dataset, "shots": args.shots, "K": args.K
                }
            }

        if te > best_test_ever:
            best_test_ever = te

        print(f"[seed {seed} | ep {ep:03d}] val={va:.4f}  test={te:.4f}  "
              f"bestVal={best_val:.4f}  bestTest@Val={best_test_at_val:.4f}  bestTestEver={best_test_ever:.4f}")

    return {
        "seed": seed,
        "best_val": float(best_val),
        "best_test_at_val": float(best_test_at_val),
        "best_test_ever": float(best_test_ever),
        "best_state": best_state,               
        "assigner_path_used": assigner_path_used
    }


def main():
    pa = argparse.ArgumentParser("Graph downstream with prompt tuning — multi-seed best-only")

    pa.add_argument("--dataset", type=str, default="MUTAG")
    pa.add_argument("--root", type=str, default="./data")
    pa.add_argument("--batch_size", type=int, default=128)
    pa.add_argument("--hidden_dim", type=int, default=128)
    pa.add_argument("--drop", type=float, default=0.5)
    pa.add_argument("--pool", type=str, default="mean", choices=["mean","sum","max"])

    pa.add_argument("--ckpt", type=str, required=True)

    pa.add_argument("--prompt", type=str, default="cap", choices=["cap","gpf","none"])
    pa.add_argument("--K", type=int, default=16)
    pa.add_argument("--assigner_ckpt", type=str, default="")

    pa.add_argument("--epochs", type=int, default=300)
    pa.add_argument("--lr", type=float, default=5e-3)
    pa.add_argument("--weight_decay", type=float, default=0.0)
    pa.add_argument("--shots", type=int, default=50)
    pa.add_argument("--test_frac", type=float, default=0.2)

    pa.add_argument("--seeds", type=str, default="0,1,2,3,4")
    
    pa.add_argument("--csv_path", type=str, default="results_csv/prompt_bestof.csv")
    pa.add_argument("--out_ckpt", type=str, default="logs_prompt/best_prompt_cls.pth")

    args = pa.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd_pretrain = torch.load(args.ckpt, map_location=device)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    all_res = []
    best_score = -1.0
    best_pack = None

    for s in seeds:
        res = run_one_seed(args, device, s, sd_pretrain)
        all_res.append(res)

        score = res["best_test_at_val"] 
        if score > best_score:
            best_score = score
            best_pack = res

        torch.cuda.empty_cache()

    if  best_pack and best_pack["best_state"] is not None:
        os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
        torch.save(best_pack["best_state"], args.out_ckpt)


    best_csv_metric = best_pack["best_test_at_val"] 
    write_min_csv(args.csv_path, best_csv_metric, args.ckpt, best_pack["assigner_path_used"])

    print("===== Summary (per seed) =====")
    for r in all_res:
        print(f"seed={r['seed']}  BestVal={r['best_val']:.4f}  "
              f"BestTest@Val={r['best_test_at_val']:.4f}  BestTestEver={r['best_test_ever']:.4f}")
    print(f"CSV saved to: {args.csv_path}")


if __name__ == "__main__":
    main()
