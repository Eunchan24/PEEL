"""
OLLM evaluation metrics (Lo et al., NeurIPS 2024).

Five metrics computed against gold taxonomy:
  1. Literal F1     - exact edge title matching
  2. Fuzzy F1       - embedding threshold matching (tau=0.436)
  3. Continuous F1  - Hungarian optimal edge matching
  4. Graph F1       - SGConv node matching (primary metric)
  5. Motif Distance - triadic census divergence (lower=better)
"""

import numpy as np
import networkx as nx
import torch
from scipy.optimize import linear_sum_assignment

EVAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FUZZY_THRESHOLD = 0.436  # OLLM default

_model_cache = {}


def _safe_f1(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# -- embedding helpers --

def _load_eval_model(model_name=None):
    if model_name is None:
        model_name = EVAL_MODEL
    if model_name in _model_cache:
        return _model_cache[model_name]

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _model_cache[model_name] = (model, tokenizer, device)
    return model, tokenizer, device


@torch.no_grad()
def _embed_texts(texts, model, tokenizer, device, batch_size=256):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True).to(device)
        out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_embs.append(emb)
    return torch.cat(all_embs, dim=0)


def _embed_graph_nodes(G, model, tokenizer, device):
    nodes_to_embed = [n for n in G.nodes if "embed" not in G.nodes[n]]
    if nodes_to_embed:
        titles = [G.nodes[n].get("title", str(n)) for n in nodes_to_embed]
        embs = _embed_texts(titles, model, tokenizer, device)
        for node, emb in zip(nodes_to_embed, embs):
            G.nodes[node]["embed"] = emb
    return G


# -- metrics --

def literal_f1(G_pred, G_gold):
    """Exact edge title matching."""
    if G_pred.number_of_edges() == 0 or G_gold.number_of_edges() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def edge_titles(G):
        return {(G.nodes[u].get("title", u), G.nodes[v].get("title", v))
                for u, v in G.edges()}

    ep, eg = edge_titles(G_pred), edge_titles(G_gold)
    matched = ep & eg
    p = len(matched) / len(ep)
    r = len(matched) / len(eg)
    return {"precision": p, "recall": r, "f1": _safe_f1(p, r)}


def fuzzy_and_continuous_f1(G_pred, G_gold, match_threshold=None, model_name=None):
    """Fuzzy F1 + Continuous F1 via embedding similarity."""
    if match_threshold is None:
        match_threshold = FUZZY_THRESHOLD

    s1, s2 = G_pred.number_of_edges(), G_gold.number_of_edges()
    if s1 == 0 or s2 == 0:
        return {"fuzzy_precision": 0.0, "fuzzy_recall": 0.0, "fuzzy_f1": 0.0,
                "continuous_precision": 0.0, "continuous_recall": 0.0, "continuous_f1": 0.0}

    model, tok, dev = _load_eval_model(model_name)
    G_pred = _embed_graph_nodes(G_pred, model, tok, dev)
    G_gold = _embed_graph_nodes(G_gold, model, tok, dev)

    def edge_embs(G, edges):
        u = torch.stack([G.nodes[a]["embed"] for a, _ in edges])
        v = torch.stack([G.nodes[b]["embed"] for _, b in edges])
        return u, v

    def cos_sim(a, b):
        return torch.mm(torch.nn.functional.normalize(a, dim=-1),
                        torch.nn.functional.normalize(b, dim=-1).t())

    ep, eg = list(G_pred.edges()), list(G_gold.edges())
    u1, v1 = edge_embs(G_pred, ep)
    u2, v2 = edge_embs(G_gold, eg)

    su, sv = cos_sim(u1, u2), cos_sim(v1, v2)

    # continuous F1: edge sim = min(source sim, target sim)
    sims = torch.minimum(su, sv).cpu().numpy()
    ri, ci = linear_sum_assignment(sims, maximize=True)
    score = sims[ri, ci].sum().item()
    cp, cr = score / s1, score / s2

    # fuzzy F1: both endpoints above threshold
    hard = ((su >= match_threshold) & (sv >= match_threshold)).cpu().numpy()
    fp = hard.any(axis=1).sum().item() / s1
    fr = hard.any(axis=0).sum().item() / s2

    return {
        "fuzzy_precision": fp, "fuzzy_recall": fr, "fuzzy_f1": _safe_f1(fp, fr),
        "continuous_precision": cp, "continuous_recall": cr, "continuous_f1": _safe_f1(cp, cr),
    }


def graph_f1(G_pred, G_gold, n_iters=2, direction="forward", model_name=None):
    """Graph F1: SGConv graph convolution + Hungarian node matching.

    This is the primary metric from OLLM (Lo et al., 2024).
    """
    import scipy.sparse as sp

    model, tok, dev = _load_eval_model(model_name)
    G_pred = _embed_graph_nodes(G_pred.copy(), model, tok, dev)
    G_gold = _embed_graph_nodes(G_gold.copy(), model, tok, dev)

    def _sgconv(G, k, direction):
        nodes = list(G.nodes())
        n = len(nodes)
        if n == 0:
            return np.zeros((0, 0))

        node_idx = {nd: i for i, nd in enumerate(nodes)}

        X = np.array([G.nodes[nd]["embed"].cpu().numpy()
                       if isinstance(G.nodes[nd].get("embed"), torch.Tensor)
                       else G.nodes[nd].get("embed", np.zeros(1))
                       for nd in nodes], dtype=np.float32)

        rows, cols = [], []
        for u, v in G.edges():
            if u in node_idx and v in node_idx:
                i, j = node_idx[u], node_idx[v]
                if direction == "forward":
                    rows.append(i); cols.append(j)
                elif direction == "reverse":
                    rows.append(j); cols.append(i)
                else:  # undirected
                    rows.extend([i, j]); cols.extend([j, i])

        if not rows:
            return X

        # self-loops
        for i in range(n):
            rows.append(i); cols.append(i)

        data = np.ones(len(rows), dtype=np.float32)
        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

        # symmetric normalization: D^{-1/2} A D^{-1/2}
        d = np.asarray(A.sum(axis=1)).flatten()
        d[d == 0] = 1.0
        D_inv = sp.diags(1.0 / np.sqrt(d))
        A_norm = D_inv @ A @ D_inv

        result = X.copy()
        for _ in range(k):
            result = A_norm @ result
        return result

    Xp = _sgconv(G_pred, n_iters, direction)
    Xg = _sgconv(G_gold, n_iters, direction)

    np_, ng = Xp.shape[0], Xg.shape[0]
    if np_ == 0 or ng == 0:
        return {"graph_precision": 0.0, "graph_recall": 0.0, "graph_f1": 0.0}

    # L2 normalize + cosine sim
    Xp = Xp / np.maximum(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-10)
    Xg = Xg / np.maximum(np.linalg.norm(Xg, axis=1, keepdims=True), 1e-10)
    sim = Xp @ Xg.T

    ri, ci = linear_sum_assignment(sim, maximize=True)
    score = sim[ri, ci].sum()
    p, r = float(score / np_), float(score / ng)

    return {"graph_precision": round(p, 4), "graph_recall": round(r, 4),
            "graph_f1": round(_safe_f1(p, r), 4)}


def motif_distance(G_pred, G_gold):
    """Triadic census divergence (Total Variation Distance)."""
    cp = nx.triadic_census(G_pred)
    cg = nx.triadic_census(G_gold)

    types = sorted(set(cp.keys()) | set(cg.keys()))
    vp = np.array([cp.get(t, 0) for t in types], dtype=float)
    vg = np.array([cg.get(t, 0) for t in types], dtype=float)

    sp, sg = vp.sum(), vg.sum()
    if sp > 0: vp /= sp
    if sg > 0: vg /= sg

    dist = float(np.abs(vp - vg).sum() / 2)
    return {"motif_distance": round(dist, 4)}


def evaluate_all(G_pred, G_gold):
    """Run all 5 OLLM metrics."""
    results = {}
    results.update(literal_f1(G_pred, G_gold))
    fc = fuzzy_and_continuous_f1(G_pred, G_gold)
    results.update(fc)
    results.update(graph_f1(G_pred, G_gold))
    results.update(motif_distance(G_pred, G_gold))
    return results
