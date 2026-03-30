"""
PEEL: Peeling Embeddings to Extract Latent ontologies.

Reproduce Table 3 from the paper:
    python run_peel.py --dataset wiki-ol
    python run_peel.py --dataset arxiv-ol --checkpoint checkpoints/wiki_ol_probe.pt
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from peel.data import load_ollm_graph, extract_edges_and_nodes, split_edges, embed_and_cache
from peel.probe import PEELProbe
from peel.train import train_probe
from peel.tree import extract_tree
from peel.dag import expand_tree_to_dag
from peel.evaluate import evaluate_all


# Section 5.1 — shared probe hyperparameters (tuned on wiki-ol val split).
# Per-dataset inference settings follow from dataset structure:
#   wiki-ol (8K nodes, depth~3): d=512 captures depth signal, k=10 kNN, DAG expansion to 15K
#   arxiv-ol (61 nodes, depth~3): full d=768, k=20 kNN (small graph), tree only (no expansion)
CONFIGS = {
    "wiki-ol": {
        "d_out": 128, "margin": 4.0, "lr": 5e-4,
        "max_epochs": 10, "batch_size": 512,
        "infer_dim": 512, "k": 10, "n_expand": 15000,
    },
    "arxiv-ol": {
        "d_out": 128, "margin": 4.0, "lr": 5e-4,
        "max_epochs": 10, "batch_size": 512,
        "infer_dim": 768, "k": 20, "n_expand": None,
    },
}


def main():
    parser = argparse.ArgumentParser(description="PEEL pipeline")
    parser.add_argument("--dataset", required=True, choices=["wiki-ol", "arxiv-ol"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to trained probe (.pt)")
    parser.add_argument("--train", action="store_true",
                        help="train probe from scratch (default: use checkpoint)")
    parser.add_argument("--dim", type=int, default=None,
                        help="override inference truncation dimension")
    parser.add_argument("--k", type=int, default=None,
                        help="override kNN candidates for CLE")
    parser.add_argument("--n-expand", type=int, default=None,
                        help="override DAG expansion budget")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset].copy()
    if args.dim is not None: cfg["infer_dim"] = args.dim
    if args.k is not None: cfg["k"] = args.k
    if args.n_expand is not None: cfg["n_expand"] = args.n_expand

    t_start = time.time()
    print(f"PEEL | dataset={args.dataset}")
    print(f"  config: dim={cfg['infer_dim']}, k={cfg['k']}, n_expand={cfg['n_expand']}")

    # 1. load data
    print("\n[1/5] Loading data...")
    gold = load_ollm_graph(args.dataset, "test")
    train_graph = load_ollm_graph(args.dataset, "train")
    edges, titles = extract_edges_and_nodes(train_graph)

    # 2. embed concepts
    print("[2/5] Embedding concepts...")
    ds_key = args.dataset.replace("-", "_")

    train_texts = [train_graph.nodes[n].get("title", str(n)) for n in train_graph.nodes()]
    train_names = [str(n) for n in train_graph.nodes()]
    train_emb, train_names = embed_and_cache(train_texts, train_names,
                                              f"nomic_{ds_key}_train")

    test_texts = [gold.nodes[n].get("title", str(n)) for n in gold.nodes()]
    test_names = [str(n) for n in gold.nodes()]
    test_emb, test_names = embed_and_cache(test_texts, test_names,
                                            f"nomic_{ds_key}_test")

    # 3. train or load probe
    print("[3/5] Probe...")
    d_in = train_emb.shape[1]

    if args.checkpoint and not args.train:
        model = PEELProbe(d_in=d_in, d_out=cfg["d_out"])
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        print(f"  loaded: {args.checkpoint}")
    elif args.train or args.checkpoint is None:
        # prepare indexed edges
        name_to_idx = {str(n): i for i, n in enumerate(train_names)}
        indexed = [(name_to_idx[str(p)], name_to_idx[str(c)])
                   for p, c in edges
                   if str(p) in name_to_idx and str(c) in name_to_idx]
        tr_edges, val_edges = split_edges(indexed, seed=args.seed)

        model, _ = train_probe(
            train_emb, tr_edges, val_edges,
            d_in=d_in, d_out=cfg["d_out"],
            lr=cfg["lr"], margin=cfg["margin"],
            batch_size=cfg["batch_size"], max_epochs=cfg["max_epochs"],
            seed=args.seed, save_dir=f"checkpoints/{ds_key}",
        )
    del train_emb; gc.collect()

    # 4. inference: truncate + CLE + DAG
    print("[4/5] Tree extraction + DAG expansion...")
    d = cfg["infer_dim"]
    emb = test_emb[:, :d]
    if d < 768:
        emb = F.pad(emb, (0, 768 - d))

    model.eval()
    tree, viol_np = extract_tree(model, emb, test_names, k_candidate=cfg["k"])
    pred = expand_tree_to_dag(tree, viol_np, test_names, cfg["n_expand"])
    del viol_np

    # attach titles for evaluation
    for node in pred.nodes():
        nid = int(node) if str(node).isdigit() else node
        if nid in gold.nodes and "title" in gold.nodes[nid]:
            pred.nodes[node]["title"] = gold.nodes[nid]["title"]
        else:
            pred.nodes[node]["title"] = str(node)

    print(f"  pred: {pred.number_of_nodes()} nodes, {pred.number_of_edges()} edges")

    # 5. evaluate
    print("[5/5] Evaluating...")
    metrics = evaluate_all(pred, gold)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Results ({args.dataset})")
    print(f"{'='*50}")
    print(f"  Lit.F1  = {metrics.get('f1', 0):.3f}")
    print(f"  Fuz.F1  = {metrics.get('fuzzy_f1', 0):.3f}")
    print(f"  Cont.F1 = {metrics.get('continuous_f1', 0):.3f}")
    print(f"  Grph.F1 = {metrics.get('graph_f1', 0):.3f}")
    print(f"  Motif.D = {metrics.get('motif_distance', 0):.3f}")
    print(f"\n  time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
