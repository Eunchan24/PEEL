"""
DAG expansion: add low-violation edges to the CLE tree while
maintaining acyclicity. Edge budget n_expand controls density.
"""

import numpy as np
import networkx as nx


def expand_tree_to_dag(tree, viol_np, node_names, target_edges):
    """Expand CLE tree into a DAG by greedily adding low-violation edges.

    Args:
        tree: nx.DiGraph (CLE arborescence)
        viol_np: (N, N) violation matrix from extract_tree
        node_names: list of node identifiers
        target_edges: total edge budget (paper: 15000 for wiki-ol)

    Returns:
        nx.DiGraph (DAG with up to target_edges edges)
    """
    if target_edges is None or target_edges <= tree.number_of_edges():
        return tree

    dag = tree.copy()
    N = len(node_names)

    # rank all non-tree pairs by violation score
    candidates = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ni, nj = node_names[i], node_names[j]
            if not dag.has_edge(ni, nj):
                candidates.append((viol_np[i, j], ni, nj))

    candidates.sort(key=lambda x: x[0])

    for score, parent, child in candidates:
        if dag.number_of_edges() >= target_edges:
            break
        dag.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(parent, child)

    return dag
