"""
OLLM benchmark data loading and embedding cache.
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"


def load_ollm_graph(dataset, split="test"):
    """Load OLLM JSON -> nx.DiGraph.

    Args:
        dataset: "wiki-ol" or "arxiv-ol"
        split: "train" or "test"
    """
    path = DATA_DIR / dataset / "train_test_split" / f"{split}_graph.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"], title=node["title"])
    for link in data["links"]:
        G.add_edge(link["source"], link["target"])
    return G


def extract_edges_and_nodes(G):
    """Return edge list and node title mapping."""
    edges = list(G.edges())
    node_titles = {n: G.nodes[n]["title"] for n in G.nodes()}
    return edges, node_titles


def split_edges(edges, seed=42, train_ratio=0.8, val_ratio=0.1):
    """Split edges into train/val/holdout."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(edges))
    n_train = int(len(edges) * train_ratio)
    n_val = int(len(edges) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    train_edges = [edges[i] for i in train_idx]
    val_edges = [edges[i] for i in val_idx]
    return train_edges, val_edges


def embed_and_cache(texts, names, cache_name,
                    model_name="nomic-ai/nomic-embed-text-v1.5",
                    prefix="search_document: ", batch_size=512):
    """Embed texts and cache to disk. Load from cache if available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}.pt"
    meta_path = CACHE_DIR / f"{cache_name}_names.json"

    if cache_path.exists() and meta_path.exists():
        print(f"  loading cache: {cache_path}")
        embeddings = torch.load(cache_path, weights_only=True)
        with open(meta_path) as f:
            cached_names = json.load(f)
        return embeddings, cached_names

    print(f"  embedding {len(texts)} texts ({model_name})...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")

    input_texts = [prefix + t for t in texts] if prefix else texts
    raw = model.encode(input_texts, batch_size=batch_size,
                       show_progress_bar=True, normalize_embeddings=False)
    embeddings = torch.from_numpy(raw).float()

    torch.save(embeddings, cache_path)
    with open(meta_path, "w") as f:
        json.dump([str(n) for n in names], f)
    print(f"  saved: {cache_path} ({embeddings.shape})")

    return embeddings, [str(n) for n in names]
