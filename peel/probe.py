"""
Order-embedding probe and violation score computation.

The probe projects frozen MRL embeddings into a non-negative order-embedding
space where coordinate-wise dominance reflects parent-child relations.
Architecture: f = ReLU(We + b), W in R^{d_out x D}  (Vendrov et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PEELProbe(nn.Module):
    """Linear probe: MRL embedding -> order-embedding space.

    With d_in=768, d_out=128: 768*128 + 128 = 98,432 parameters.
    """

    def __init__(self, d_in=768, d_out=128):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=True)
        nn.init.normal_(self.proj.weight, std=0.01)  # Ganea et al. (2018)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        return F.relu(self.proj(x))


def order_loss(model, parent_emb, child_emb, neg_emb, margin=4.0):
    """Order-embedding loss (Vendrov et al., ICLR 2016).

    L+ encourages f_parent <= f_child coordinate-wise.
    L- pushes negatives apart by at least margin.
    """
    p = model(parent_emb)
    c = model(child_emb)
    n = model(neg_emb)

    # L+: ||max(0, f_p - f_c)||^2
    pos_loss = F.relu(p - c).norm(dim=-1).pow(2)

    # L-: max(0, m - ||max(0, f_p - f_n)||^2)
    neg_energy = F.relu(p - n).norm(dim=-1).pow(2)
    neg_loss = F.relu(margin - neg_energy)

    return pos_loss.mean() + neg_loss.mean()


def compute_pairwise_violations(model, embeddings, chunk_size=512):
    """Compute (N, N) violation matrix.

    violations[i,j] = ||max(0, f(i) - f(j))||^2
    Low v(i,j) => i is likely parent of j.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))

    with torch.no_grad():
        model_dev = model.to(device)
        tau = model_dev(embeddings.to(device))

        N = tau.shape[0]
        violations = torch.zeros(N, N)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            diff = tau[start:end].unsqueeze(1) - tau.unsqueeze(0)
            chunk_viol = torch.clamp(diff, min=0).pow(2).sum(-1)
            violations[start:end] = chunk_viol.cpu()
            del diff, chunk_viol

        model.to("cpu")
    return violations
