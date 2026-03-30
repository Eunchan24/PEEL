# PEEL: Peeling Embeddings to Extract Latent Ontologies

> Peeling the Matryoshka: Extracting Ontological Structure from Retrieval Embeddings

## Overview

PEEL extracts taxonomy DAGs from frozen Matryoshka Representation Learning (MRL) embeddings. It trains a 98K-parameter order-embedding probe on gold parent-child edges, then extracts a tree via Chu-Liu/Edmonds minimum spanning arborescence and optionally expands it into a DAG.

## Setup

```bash
pip install -r requirements.txt
```

For faster CLE tree extraction (optional):
```bash
pip install ufal.chu_liu_edmonds
```

## Data

PEEL uses the OLLM benchmarks (Lo et al., 2024). Download the datasets from the [OLLM repository](https://github.com/andylolu2/ollm) and place them as:

```
data/
├── wiki-ol/
│   └── train_test_split/
│       ├── train_graph.json
│       └── test_graph.json
└── arxiv-ol/
    └── train_test_split/
        ├── train_graph.json
        └── test_graph.json
```

## Usage

Reproduce Table 3 (wiki-ol) using the pre-trained probe:
```bash
python run_peel.py --dataset wiki-ol --checkpoint checkpoints/wiki_ol_probe.pt
```

Train from scratch (~4 seconds on a single GPU):
```bash
python run_peel.py --dataset wiki-ol --train
```

Zero-shot transfer to arXiv-ol (no arXiv training data used):
```bash
python run_peel.py --dataset arxiv-ol --checkpoint checkpoints/wiki_ol_probe.pt
```

## Method

```
peel/
├── probe.py      §4.1  Order-Embedding Probe (768→128, ReLU)
├── train.py      §4.2  Training with order-embedding loss
├── tree.py       §4.3  CLE tree extraction
├── dag.py        §4.4  DAG expansion
├── data.py             Data loading (OLLM benchmarks)
└── evaluate.py         Evaluation (5 OLLM metrics)
```

## Hyperparameters

| Parameter | wiki-ol | arXiv-ol |
|-----------|---------|----------|
| Probe     | 768→128, ReLU | same (zero-shot) |
| Margin    | 4.0     | —        |
| LR        | 5e-4    | —        |
| Epochs    | 10      | —        |
| Infer dim | 512     | 768      |
| CLE k     | 10      | 20       |
| n_expand  | 15,000  | tree only |

## Reproducibility

All hyperparameters were tuned on a 10% validation split of wiki-ol training edges, never on the test set. PEEL follows the OLLM evaluation protocol (Lo et al., 2024) exactly: identical concept vocabulary, train/test splits, and five metrics. The probe is deterministic given seed=42; training takes ~4 seconds on a single GPU.

## License

Apache-2.0
