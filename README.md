# Universal Adaptation of Graph Neural Networks via Class-Aware Prompting

This repository contains the anonymous implementation of the paper **"Universal Adaptation of Graph Neural Networks via Class-Aware Prompting"**, submitted to **ICML 2026**.

## 🚀 Introduction

**Abstract:**
Graph prompting has emerged as a parameter-efficient paradigm for adapting pretrained Graph Neural Networks (GNNs). However, existing strategies predominantly employ class-agnostic designs, typically applying uniform global prompts or feature-dependent attention. These approaches are often insufficient, as node labels are intrinsically determined by the coupling of node features and graph topology, rather than features alone. To bridge this gap, we propose CAP (\underline{C}lass-\underline{A}ware \underline{P}rompting) for universal GNN adaptation. CAP operates by inferring latent class memberships from the coupling of node features and topology, subsequently retrieving compositional prompts from a learnable dictionary of class-prototype tokens. By injecting these contextualized semantic priors into the input space, CAP enriches the input features with the necessary context to resolve semantic ambiguities. Crucially, this node-level sharpening extends to graph-level tasks by yielding high-fidelity embeddings that result in separable global representations after pooling. Experiments demonstrate that CAP achieves state-of-the-art performance on few-shot benchmarks, exhibiting universal robustness across both homophilous and heterophilous graphs.

## 🛠️ Dependencies

The code is built with **Python 3.9** and **PyTorch 1.12+**.

* `numpy>=1.20`
* `torch>=1.12.1`
* `torch-geometric>=2.5.0`
* `ogb>=1.3.6`
* `scikit-learn`
* `tqdm`

## 💻 Usage

### 1. Installation

We recommend using Anaconda to manage the environment.

# Create environment
conda create -n cap_gnn python=3.9
conda activate cap_gnn

# Install PyTorch (adjust cuda version to match your system)
# Example for CUDA 11.3:
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install Graph Dependencies
pip install torch-geometric
pip install scikit-learn tqdm ogb

### 2. Run Code

#### 🟢 For Graph Classification

The script `graph_task.py` handles graph-level tasks.

**Example: Run CAP on NCI109 (50-shot)**

Bash
# Modify parameters in the command line as needed
python graph_task.py \
  --dataset MUTAG \
  --ckpt "model/MUTAG_edgepred_hid128.pth.pth" \
  --assigner_ckpt "model/MUTAG_edgepred_hid128.pth.pth" \
  --prompt cap \
  --K 8 \
  --shots 50 \
  --batch_size 32 \
  --lr 0.005 \
  --test_frac 0.4
#### For Node Classification

The script `node_task.py` handles node-level tasks.

**Example: Run CAP on Cora (5-shot)**

Bash
# This script supports grid search over hyperparameters
python node_task.py \
  --dataset Cora \
  --ckpt model/Cora_edgepred_hid128.pth\
  --shots 5 \
  --ssl graphcl \
  --hidden_dim 128 \
  --gpu 0\
  --K 16

## Parameters

Below are the common arguments used in the training scripts:

| **Parameter**     | **Description**                                       | **Default**               |
| ----------------- | ----------------------------------------------------- | ------------------------- |
| `--dataset`       | Target dataset name (e.g., `Cora`, `NCI109`, `MUTAG`) | `MUTAG` / `Cora`          |
| `--shots`         | Number of samples per class for few-shot learning     | `50` (Graph) / `5` (Node) |
| `--ckpt`          | Path to the pre-trained GNN model checkpoint          | Required                  |
| `--assigner_ckpt` | Path to the assigner checkpoint                       | `""`                      |
| `--prompt`        | Prompting method to use (`cap`, `gpf`, `none`)        | `cap`                     |
| `--K`             | Number of clusters/prototypes for the assigner        | `16`                      |
| `--lr`            | Learning rate for the prompt tuning                   | `0.005`                   |
| `--hidden_dim`    | Hidden dimension size of the GNN                      | `128`                     |
| `--batch_size`    | Batch size for training                               | `128`                     |
| `--seeds`         | List of random seeds for reproducibility              | `0,1,2,3,4`               |
