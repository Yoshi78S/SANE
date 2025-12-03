# SANE: Structure-Aware Negative Estimation for Recommendation

This repository contains the PyTorch implementation of **SANE**, a recommendation model that adaptively weights negative feedback based on graph structure information.

## Overview

Traditional recommendation models with negative feedback treat all negative items equally. SANE addresses this limitation by:

1. **Adaptive Negative Weighting**: Utilizing Neighborhood Similarity (Nsim) to estimate the reliability of negative feedback
2. **Graph Denoising**: Dynamically removing noisy edges based on structural information
3. **Item-Pair Contrastive Learning**: Pulling similar items together and pushing dissimilar items apart

### Key Innovation

SANE computes **Nsim (Neighborhood Similarity)** using staggered layer embeddings:

```
Nsim = 0.5 * (e1_u · e2_i + e2_u · e1_i)
```

Where `e1` and `e2` are embeddings from different GNN layers. This captures structural consistency:
- **High Nsim for positive edges** → Confident positive interaction
- **Low Nsim for negative edges** → Strong negative signal
- **Low Nsim for positive edges** → Potentially noisy positive
- **High Nsim for negative edges** → Potentially false negative

## Environment

```
python==3.9.19
numpy==1.24.4
pandas==2.0.3
scipy==1.11.4
torch==2.0.1
torch-geometric==2.3.1
torchsparsegradutils==0.1.2
```

### Installation

```bash
# Create conda environment
conda create -n sane python=3.9
conda activate sane

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1

# Install other dependencies
pip install torch-geometric==2.3.1
pip install torchsparsegradutils==0.1.2
pip install numpy==1.24.4 pandas==2.0.3 scipy==1.11.4
```

## Datasets

| Dataset | #Users | #Items | #Interactions | Pos/Neg |
|---------|--------|--------|---------------|---------|
| Amazon-CDs | 51,267 | 46,464 | 895,266 | 1:0.22 |
| Amazon-Music | 3,472 | 2,498 | 49,875 | 1:0.25 |
| Epinions | 17,894 | 17,660 | 413,774 | 1:0.37 |
| KuaiRec | 1,411 | 3,327 | 253,983 | 1:5.95 |
| KuaiRand | 16,974 | 4,373 | 263,100 | 1:1.25 |

## Usage

### Basic Training

```bash
python -u code/main.py --data=amazon-music --offset=4 --alpha=0.0 --beta=1.0 --sample_hop=3
```

### Training with SANE (Adaptive Negative Weighting + Graph Denoising)

```bash
python -u code/main.py --data=amazon-music --offset=4 --alpha=0.0 --beta=1.0 --sample_hop=3 \
  --niden_start_epoch=150 \
  --niden_rate_max=0.02 \
  --niden_rate_min=0.01 \
  --niden_decay_rate=0.0 \
  --niden_update_interval=5
```

### Training with Item-Pair Contrastive Learning

```bash
python -u code/main.py --data=amazon-music --offset=4 --alpha=0.0 --beta=1.0 --sample_hop=3 \
  --item_pair_weight=0.005 \
  --item_pair_tau=0.05 \
  --item_pair_sample_size=100000
```

## Parameters

### Base Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | `epinions` | Dataset name |
| `--offset` | `4.0` | Threshold for positive/negative classification |
| `--alpha` | `0.0` | Weight for negative graph in Laplacian |
| `--beta` | `1.0` | Weight for negative samples in BPR loss |
| `--sample_hop` | `4` | Number of hops for graph sampling |
| `--n_layers` | `3` | Number of GNN layers |
| `--hidden_dim` | `64` | Embedding dimension |

### SANE Parameters (Adaptive Negative Weighting)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--niden_start_epoch` | `1500` | Epoch to start adaptive weighting |
| `--niden_rate_max` | `0.01` | Maximum edge removal rate |
| `--niden_rate_min` | `0.005` | Minimum edge removal rate |
| `--niden_decay_rate` | `0.0` | Rate change per epoch |
| `--niden_update_interval` | `5` | Epochs between graph updates |

### Item-Pair Contrastive Learning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--item_pair_weight` | `0.005` | Weight for contrastive loss |
| `--item_pair_tau` | `0.05` | Temperature for InfoNCE |
| `--item_pair_sample_size` | `100000` | Number of pairs per epoch |

## Project Structure

```
SANE/
├── code/
│   ├── main.py          # Main training script
│   ├── model.py         # SANE model implementation
│   ├── dataloader.py    # Dataset and graph construction
│   ├── parse.py         # Argument parser
│   └── utils.py         # Evaluation metrics
├── data/
│   ├── amazon-cds/
│   ├── amazon-music/
│   ├── epinions/
│   ├── KuaiRec/
│   └── KuaiRand/
└── results/             # Training logs
```

## Method Details

### 1. Adaptive Negative Weighting in BPR Loss

After `niden_start_epoch`, the BPR loss weights samples by Nsim:
- **Positive samples**: `weight = Nsim / max(Nsim)` (high confidence = high weight)
- **Negative samples**: `weight = β × (1 - Nsim) / max(1 - Nsim)` (low Nsim = strong negative)

### 2. Graph Denoising

Edges are masked based on Nsim thresholds:
- **Positive edges**: Mask if `Nsim < λ_pos` (potentially noisy)
- **Negative edges**: Mask if `Nsim ≥ λ_neg` (potentially false negatives)

### 3. Item-Pair Contrastive Learning

InfoNCE loss on item pairs:
- **(+, +) pairs**: Items liked by the same user → pull together
- **(+, -) pairs**: Positive and negative items of the same user → push apart

## Acknowledgements

This implementation is based on [SIGformer](https://github.com/StupidThree/SIGformer) and incorporates ideas from NiDen for graph denoising.

