"""
Training loop for the PEEL order-embedding probe.

Supports optional random truncation (Section 4.2): at each training step,
randomly sample d from {64,128,256,384,512,768}, truncate embeddings,
and zero-pad back to D=768.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .probe import PEELProbe, order_loss


def _sample_negatives(parent_ids, child_ids, n_nodes, parent_to_children,
                      neg_ratio=10, rng=None):
    """Negative sampling: 50% random + 30% sibling + 20% reversed."""
    if rng is None:
        rng = np.random.RandomState()

    batch_size = len(parent_ids)
    n_random = int(neg_ratio * 0.5)
    n_sibling = int(neg_ratio * 0.3)
    n_reverse = neg_ratio - n_random - n_sibling

    all_negs = []
    for i in range(batch_size):
        pid, cid = int(parent_ids[i]), int(child_ids[i])
        negs = list(rng.randint(0, n_nodes, size=n_random))

        siblings = parent_to_children.get(pid, [])
        if len(siblings) > 1:
            sibs = [s for s in siblings if s != cid]
            if sibs:
                chosen = rng.choice(sibs, size=min(n_sibling, len(sibs)),
                                    replace=True)
                negs.extend(chosen.tolist())
        while len(negs) < n_random + n_sibling:
            negs.append(rng.randint(0, n_nodes))

        negs.extend([cid] * n_reverse)
        all_negs.extend(negs)

    return np.array(all_negs, dtype=np.int64)


def train_probe(
    embeddings,
    train_edges,
    val_edges,
    d_in=768,
    d_out=128,
    lr=5e-4,
    margin=4.0,
    batch_size=512,
    max_epochs=10,
    patience=5,
    neg_ratio=10,
    seed=42,
    truncation_dims=None,
    save_dir=None,
):
    """Train the PEEL probe with order-embedding loss.

    Args:
        embeddings: (N, D) pre-computed MRL embeddings
        train_edges: [(parent_idx, child_idx), ...]
        val_edges: [(parent_idx, child_idx), ...]
        truncation_dims: if set, enables random truncation training (Section 4.2)
                         e.g. [64, 128, 256, 384, 512, 768]
        save_dir: save model checkpoint and training log

    Returns:
        (trained_model, history_dict)
    """
    import random as py_random

    rng = np.random.RandomState(seed)
    py_rng = py_random.Random(seed)
    torch.manual_seed(seed)
    n_nodes = embeddings.shape[0]
    d_full = embeddings.shape[1]

    parent_to_children = {}
    for p, c in train_edges:
        parent_to_children.setdefault(p, []).append(c)

    model = PEELProbe(d_in=d_in, d_out=d_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  PEELProbe({d_in}->{d_out}) = {n_params:,} params")
    print(f"  margin={margin}, lr={lr}, epochs={max_epochs}")
    if truncation_dims:
        print(f"  random truncation: {truncation_dims}")

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    train_edges_np = np.array(train_edges)
    history = []

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(train_edges_np))
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            batch = train_edges_np[idx]
            p_ids, c_ids = batch[:, 0], batch[:, 1]

            neg_ids = _sample_negatives(p_ids, c_ids, n_nodes,
                                        parent_to_children, neg_ratio, rng)
            p_rep = np.repeat(p_ids, neg_ratio)
            c_rep = np.repeat(c_ids, neg_ratio)

            p_emb = embeddings[p_rep]
            c_emb = embeddings[c_rep]
            n_emb = embeddings[neg_ids]

            # random truncation (Section 4.2)
            if truncation_dims:
                d = py_rng.choice(truncation_dims)
                p_emb = p_emb[:, :d]
                c_emb = c_emb[:, :d]
                n_emb = n_emb[:, :d]
                if d < d_full:
                    p_emb = F.pad(p_emb, (0, d_full - d))
                    c_emb = F.pad(c_emb, (0, d_full - d))
                    n_emb = F.pad(n_emb, (0, d_full - d))

            loss = order_loss(model, p_emb, c_emb, n_emb, margin=margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # validation
        val_loss, val_auroc = _validate(model, embeddings, val_edges, margin)
        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        mark = " *" if wait == 0 else ""
        print(f"  epoch {epoch:3d} | loss={avg_loss:.4f} "
              f"| val_loss={val_loss:.4f} | auroc={val_auroc:.4f} "
              f"| {elapsed:.1f}s{mark}")

        history.append({
            "epoch": epoch, "train_loss": avg_loss,
            "val_loss": val_loss, "val_auroc": val_auroc,
        })

        if wait >= patience:
            print(f"  early stop (best={best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        with open(save_dir / "train_log.json", "w") as f:
            json.dump({"best_epoch": best_epoch, "history": history}, f, indent=2)
        print(f"  saved: {save_dir / 'model.pt'}")

    return model, {"best_epoch": best_epoch, "history": history}


@torch.no_grad()
def _validate(model, embeddings, val_edges, margin):
    model.eval()
    val_np = np.array(val_edges)
    p_emb, c_emb = embeddings[val_np[:, 0]], embeddings[val_np[:, 1]]

    p_proj, c_proj = model(p_emb), model(c_emb)
    pos_energy = F.relu(p_proj - c_proj).norm(dim=-1).pow(2)

    rng = np.random.RandomState(9999)
    n_emb = embeddings[rng.randint(0, embeddings.shape[0], size=len(val_edges))]
    n_proj = model(n_emb)
    neg_energy = F.relu(p_proj - n_proj).norm(dim=-1).pow(2)

    val_loss = (pos_energy.mean() + F.relu(margin - neg_energy).mean()).item()

    labels = np.concatenate([np.ones(len(pos_energy)), np.zeros(len(neg_energy))])
    scores = np.concatenate([-pos_energy.cpu().numpy(), -neg_energy.cpu().numpy()])
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5

    return val_loss, auroc
