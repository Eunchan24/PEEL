"""
CLE tree extraction via Chu-Liu/Edmonds minimum spanning arborescence.

Given violation scores from the trained probe, extract an optimal rooted
tree with exactly N-1 edges. Supports ufal C++ backend (fast) with
NetworkX fallback.
"""

import gc
import numpy as np
import networkx as nx
import torch

from .probe import PEELProbe, compute_pairwise_violations

try:
    from ufal.chu_liu_edmonds import chu_liu_edmonds as _ufal_cle
    HAS_UFAL = True
except ImportError:
    HAS_UFAL = False


def _msa_ufal(viol_np, node_names, k_candidate, root_weight):
    """MSA via ufal.chu_liu_edmonds (C++ backend).

    ufal uses scores[dep][head] convention and finds maximum spanning
    arborescence, so we negate violations (minimize -> maximize).
    """
    N = len(node_names)
    scores = np.full((N + 1, N + 1), -1e18, dtype=np.float64)

    # root -> each node
    for j in range(N):
        scores[j + 1][0] = -root_weight

    # real edges with kNN pruning
    for j in range(N):
        col = viol_np[:, j].copy()
        col[j] = np.inf
        k = min(k_candidate, N - 1)
        top_parents = np.argpartition(col, k)[:k]
        for i in top_parents:
            scores[j + 1][i + 1] = -float(col[i])

    heads, _ = _ufal_cle(scores)

    arb = nx.DiGraph()
    for j in range(1, N + 1):
        parent_idx = heads[j]
        if parent_idx == 0:
            continue
        if parent_idx >= 1:
            arb.add_edge(node_names[parent_idx - 1], node_names[j - 1])

    return arb


def _msa_networkx(viol_np, node_names, k_candidate, root_weight):
    """MSA via NetworkX (Python fallback)."""
    N = len(node_names)
    G = nx.DiGraph()

    for j in range(N):
        col = viol_np[:, j].copy()
        col[j] = float("inf")
        k = min(k_candidate, N - 1)
        top_parents = np.argpartition(col, k)[:k]
        for i in top_parents:
            G.add_edge(node_names[i], node_names[j], weight=float(col[i]))

    root = "__ROOT__"
    for n in node_names:
        G.add_edge(root, n, weight=root_weight)

    arb = nx.minimum_spanning_arborescence(G)
    if root in arb:
        arb.remove_node(root)
    return arb


def extract_tree(model, embeddings, node_names, k_candidate=100, use_ufal=True):
    """Extract CLE tree from trained probe.

    Args:
        model: trained PEELProbe
        embeddings: (N, D) tensor
        node_names: list of node identifiers
        k_candidate: kNN candidates per node (paper: k=10 wiki, k=20 arxiv)
        use_ufal: try C++ backend first

    Returns:
        (nx.DiGraph, np.ndarray): tree with N-1 edges, and (N,N) violation matrix
    """
    N = len(node_names)
    assert embeddings.shape[0] == N

    violations = compute_pairwise_violations(model, embeddings)
    viol_np = violations.cpu().numpy()
    del violations; gc.collect()

    # root weight = 95th percentile of positive violations
    pos = viol_np[viol_np > 0]
    root_weight = float(np.percentile(pos, 95)) if len(pos) > 0 else 1.0

    if use_ufal and HAS_UFAL:
        try:
            arb = _msa_ufal(viol_np, node_names, k_candidate, root_weight)
        except Exception as e:
            print(f"  ufal failed: {e}. Falling back to NetworkX.")
            arb = _msa_networkx(viol_np, node_names, k_candidate, root_weight)
    else:
        if use_ufal and not HAS_UFAL:
            print("  ufal not installed. Using NetworkX.")
        arb = _msa_networkx(viol_np, node_names, k_candidate, root_weight)

    return arb, viol_np
