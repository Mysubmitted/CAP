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

```bash
# Create environment
conda create -n cap_gnn python=3.9
conda activate cap_gnn

# Install PyTorch (adjust cuda version to match your system)
# Example for CUDA 11.3:
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install Graph Dependencies
pip install torch-geometric
pip install scikit-learn tqdm ogb
