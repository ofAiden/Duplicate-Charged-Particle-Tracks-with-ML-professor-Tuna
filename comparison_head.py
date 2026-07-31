"""
Learnable pair comparison head -- replaces the fixed-distance choice entirely.

WHY
---
DeepMatcher (Mudgal et al., SIGMOD 2018), the reference study for entity
matching, reports flatly: "We find that fixed distance functions perform poorly
overall", and that an element-wise absolute difference beats concatenation by up
to 40% relative F1. The whole Euclidean-vs-Mahalanobis-vs-cosine axis this
project has been arguing about is the axis that study found least productive --
and `verify_gauge_freedom.py` shows two of the three are the same model anyway.

A learnable head sidesteps the question. Instead of picking a distance, feed
SYMMETRIC pair features into a small MLP and let it learn the comparison.

SYMMETRY BY CONSTRUCTION
------------------------
Duplicate detection is a symmetric relation: "is a the same particle as b" must
equal "is b the same particle as a". Every feature block used here is symmetric
under swapping a and b:

    diff2 = (z_a - z_b)^2        prod  = z_a * z_b
    sq_sum = z_a^2 + z_b^2       absd  = |z_a - z_b|

so s(a,b) == s(b,a) exactly, for any weights. Note the deliberate contrast with
`arbitration.py`, where the ARBITER is antisymmetric by construction because
"which one do I keep" must flip when the pair is reordered. Detector symmetric,
arbiter antisymmetric -- both enforced architecturally rather than learned.

WHAT IS AND IS NOT EXACTLY REPRESENTABLE  (measured in _verify_containment)
--------------------------------------------------------------------------
With the diff2 block a linear readout gives EXACTLY:
  * squared Euclidean   (unit weights)
  * DIAGONAL Mahalanobis (weights = diag(V))
It does NOT give a full Mahalanobis, which needs the cross terms
diff_i * diff_j (i != j) -- those are the off-diagonal of an outer product and
are absent from an element-wise feature set. Cosine needs a ratio of the dot
product to the norms, which is not linear in any of these blocks either.

So "strictly contains all three metrics" -- which is how an earlier research pass
phrased it -- is FALSE as an exact statement. The honest claim is: it exactly
contains squared Euclidean and diagonal Mahalanobis, and approximates full
Mahalanobis and cosine to whatever accuracy the MLP achieves. The numbers below
measure that, rather than assuming it.

Run:  .venv/bin/python comparison_head.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCKS = ("diff2", "prod", "sq_sum", "absd")


def pair_features(z_a, z_b, blocks=BLOCKS, include_outer=False):
    """Symmetric pair features. Every block is invariant to swapping a and b.

    `include_outer` adds the upper-triangular cross terms of the outer product
    of the difference, which is what a FULL Mahalanobis needs. It costs
    emb_dim*(emb_dim-1)/2 extra inputs (66 at emb_dim=12), so it is off by
    default and exists to test whether those terms actually buy anything.
    """
    feats = []
    diff = z_a - z_b
    if "diff2" in blocks:
        feats.append(diff ** 2)
    if "prod" in blocks:
        feats.append(z_a * z_b)
    if "sq_sum" in blocks:
        feats.append(z_a ** 2 + z_b ** 2)
    if "absd" in blocks:
        feats.append(diff.abs())
    if include_outer:
        d = diff.shape[1]
        iu, ju = torch.triu_indices(d, d, offset=1)
        feats.append(diff[:, iu] * diff[:, ju])
    return torch.cat(feats, dim=1)


def n_features(emb_dim, blocks=BLOCKS, include_outer=False):
    n = emb_dim * len(blocks)
    if include_outer:
        n += emb_dim * (emb_dim - 1) // 2
    return n


class ComparisonHead(nn.Module):
    """Symmetric learnable comparison. Outputs a logit; higher = more likely
    the same particle."""

    def __init__(self, emb_dim, hidden=64, blocks=BLOCKS, include_outer=False,
                 depth=2):
        super().__init__()
        self.blocks = blocks
        self.include_outer = include_outer
        d_in = n_features(emb_dim, blocks, include_outer)
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        self.d_in = d_in

    def forward(self, z_a, z_b):
        return self.mlp(pair_features(z_a, z_b, self.blocks, self.include_outer))


class LinearReadout(nn.Module):
    """A single linear layer on the pair features -- the minimal head, used to
    test what is EXACTLY representable without any nonlinearity."""

    def __init__(self, emb_dim, blocks=("diff2",), include_outer=False):
        super().__init__()
        self.blocks, self.include_outer = blocks, include_outer
        self.lin = nn.Linear(n_features(emb_dim, blocks, include_outer), 1)

    def forward(self, z_a, z_b):
        return self.lin(pair_features(z_a, z_b, self.blocks, self.include_outer))


# ----------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------
def _fit(model, z_a, z_b, target, steps=3000, lr=1e-2):
    """Fit a head to reproduce a target score. Returns normalised MSE."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t = (target - target.mean()) / target.std()
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(model(z_a, z_b), t)
        loss.backward()
        opt.step()
        sched.step()
    return float(loss.detach())


def _verify_containment(n=8192, emb=12, seed=0):
    torch.manual_seed(seed)
    z_a, z_b = torch.randn(n, emb), torch.randn(n, emb)

    A = torch.randn(emb, emb)
    V_full = A.T @ A + 0.05 * torch.eye(emb)
    V_diag = torch.diag(torch.rand(emb) * 3 + 0.2)

    diff = z_a - z_b
    targets = {
        "squared Euclidean": (diff ** 2).sum(1, keepdim=True),
        "diagonal Mahalanobis": ((diff @ V_diag) * diff).sum(1, keepdim=True),
        "FULL Mahalanobis": ((diff @ V_full) * diff).sum(1, keepdim=True),
        "cosine distance": (1 - F.cosine_similarity(z_a, z_b, dim=1)).view(-1, 1),
    }

    print("=" * 78)
    print("WHAT CAN THE PAIR-FEATURE HEAD ACTUALLY REPRESENT?")
    print("=" * 78)
    print("  Normalised MSE fitting each target. 0 = exact, 1 = no better than the mean.\n")
    print(f"  {'target':<24}{'linear(diff2)':>15}{'linear(+outer)':>16}{'MLP(all)':>11}")

    for name, tgt in targets.items():
        torch.manual_seed(1)
        r_lin = _fit(LinearReadout(emb, ("diff2",)), z_a, z_b, tgt)
        torch.manual_seed(1)
        r_out = _fit(LinearReadout(emb, ("diff2",), include_outer=True), z_a, z_b, tgt)
        torch.manual_seed(1)
        r_mlp = _fit(ComparisonHead(emb, hidden=64), z_a, z_b, tgt)
        print(f"  {name:<24}{r_lin:>15.6f}{r_out:>16.6f}{r_mlp:>11.6f}")

    print("\n  Reading: a linear readout on diff2 reproduces squared Euclidean and")
    print("  DIAGONAL Mahalanobis exactly, but not a FULL Mahalanobis -- adding the")
    print("  outer-product cross terms fixes that, confirming the missing capacity")
    print("  is precisely the off-diagonal. Cosine needs a ratio, so no linear")
    print("  readout reaches it and only the MLP approximates it.")


def _verify_symmetry(n=4096, emb=12):
    print("\n" + "=" * 78)
    print("IS THE HEAD SYMMETRIC BY CONSTRUCTION?")
    print("=" * 78)
    torch.manual_seed(0)
    z_a, z_b = torch.randn(n, emb), torch.randn(n, emb)
    for outer in (False, True):
        h = ComparisonHead(emb, include_outer=outer)
        with torch.no_grad():
            err = (h(z_a, z_b) - h(z_b, z_a)).abs().max()
        tag = "with outer terms" if outer else "default blocks"
        print(f"  {tag:<20} max |s(a,b) - s(b,a)| = {float(err):.3e}"
              f"   {'OK' if err < 1e-6 else '** ASYMMETRIC **'}")
    print("  Contrast arbitration.py, where antisymmetry is enforced instead --")
    print("  detector symmetric, arbiter antisymmetric, both architectural.")


def _one_task(name, y_match, z_a, z_b, emb, g, n):
    """Score a labelling with fixed metrics, a fitted full quadratic form, and
    the MLP. The fitted quadratic form is the honest baseline: it is the best
    possible global Mahalanobis, cross terms included."""
    from sklearn.metrics import roc_auc_score

    idx = torch.randperm(n, generator=g)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    yt = y_match.numpy().ravel()
    ten = te.numpy()
    diff = z_a - z_b

    rows = {
        "Euclidean": -(diff ** 2).sum(1).numpy(),
        "cosine": F.cosine_similarity(z_a, z_b, dim=1).numpy(),
    }

    def train(model, steps, lr):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(z_a[tr], z_b[tr]), y_match[tr])
            loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            return model(z_a, z_b).squeeze(-1).numpy()

    torch.manual_seed(1)
    rows["Mahalanobis (V fitted, full)"] = train(
        LinearReadout(emb, ("diff2",), include_outer=True), 3000, 1e-2)
    torch.manual_seed(1)
    rows["learnable head (MLP)"] = train(ComparisonHead(emb, hidden=64, depth=2),
                                         3000, 3e-3)

    print(f"\n  --- {name} | positive rate {float(y_match.mean()):.3f} ---")
    aucs = {}
    for k, v in rows.items():
        aucs[k] = roc_auc_score(yt[ten], v[ten])
        print(f"    {k:<30}{aucs[k]:>10.4f}")
    gap = aucs["learnable head (MLP)"] - aucs["Mahalanobis (V fitted, full)"]
    print(f"    {'MLP - fitted quadratic':<30}{gap:>+10.4f}")
    return gap


def _verify_beats_fixed(n=20000, emb=12, seed=0):
    """Does a learnable head actually beat the best FIXED-FORM metric?

    Two probes. The first is a conjunctive rule (agree on all of a few
    coordinates); a quadratic form approximates it well by concentrating weight
    on those coordinates, so the expected gain is small. The second is a
    DISJUNCTIVE rule (agree on coordinate 0 OR coordinate 1) -- a single
    quadratic form cannot represent a union of two slabs, so this is where the
    families genuinely separate.

    Reporting both matters: the honest conclusion depends on which regime the
    real physics is in, and that is an empirical question about LST data, not
    something to be settled by picking the flattering probe.
    """
    print("\n" + "=" * 78)
    print("DOES A LEARNABLE HEAD BEAT THE BEST FIXED-FORM METRIC?")
    print("=" * 78)
    g = torch.Generator().manual_seed(seed)
    z_a = torch.randn(n, emb, generator=g)
    z_b = torch.randn(n, emb, generator=g)
    d = (z_a - z_b).abs()

    # conjunctive: close on ALL of coords 0,1,2
    y_conj = (d[:, [0, 1, 2]].max(dim=1).values < 0.8).float().view(-1, 1)
    # disjunctive: close on coord 0 OR coord 1 (a union of two slabs)
    y_disj = ((d[:, 0] < 0.25) | (d[:, 1] < 0.25)).float().view(-1, 1)

    gaps = {
        "conjunctive (AND over 3 coords)":
            _one_task("conjunctive (AND over 3 coords)", y_conj, z_a, z_b, emb,
                      torch.Generator().manual_seed(seed), n),
        "disjunctive (OR over 2 coords)":
            _one_task("disjunctive (OR over 2 coords)", y_disj, z_a, z_b, emb,
                      torch.Generator().manual_seed(seed), n),
    }

    print("\n  HONEST READING:")
    for k, v in gaps.items():
        verdict = ("material" if v > 0.02 else
                   "marginal -- do not claim a win" if v > 0.005 else
                   "no advantage")
        print(f"    {k:<34} {v:+.4f}  ({verdict})")
    print("\n  The learnable head's case does NOT rest on beating a tuned quadratic")
    print("  form everywhere -- on conjunctive structure it barely does. It rests on")
    print("  (a) removing the metric choice entirely, which verify_gauge_freedom.py")
    print("  shows is partly illusory anyway, and (b) covering disjunctive structure")
    print("  that no quadratic form can express. Whether LST duplicates are")
    print("  conjunctive or disjunctive in the learned embedding is measurable on")
    print("  real data and currently unknown.")


if __name__ == "__main__":
    _verify_containment()
    _verify_symmetry()
    _verify_beats_fixed()
