"""
Learned arbitration for LST duplicate removal.

THE GAP
-------
The embedding answers a SYMMETRIC question -- "are these two candidates the
same particle?" -- but duplicate removal requires an ASYMMETRIC one: given that
they are the same particle, which one do you keep?

Production CMSSW answers the second question with a hard-coded collection
priority (pT5 > pT3 > T5 > pLS) plus a shared-hit veto; the learned embedding
contributes nothing to it (RecoTracker/LSTCore/src/alpaka/TrackCandidate.h,
`CrossCleanT5` / `CrossCleanpLS`).  The only published learned alternative in
HEP tracking (Allaire et al., arXiv:2312.05070, ACTS) learns an intra-cluster
ranking but needs SHARED HITS to form its clusters and uses post-fit features
(chi2/NDF, holes, outliers) that do not exist at LST's pre-fit stage -- so it
cannot arbitrate cross-collection pLS-vs-T5 duplicates, which is precisely
LST's dominant residual duplicate population.

This is structurally the problem computer vision solved when it replaced
non-maximum suppression: NMS is a hand-coded priority rule (keep the highest
score, delete the overlapping) applied to a symmetric IoU relation.  The
replacements were GossipNet (arXiv:1705.02950, a network over pairwise
detection features), Relation Networks (arXiv:1711.11575, attention over the
detection set), and DETR (arXiv:2005.12872, one-to-one Hungarian matching that
makes duplicates structurally impossible).

Two heads are provided:

  PairwiseArbiter  -- for a duplicate pair, score which candidate to keep.
                      Antisymmetry s(i,j) = -s(j,i) is enforced BY
                      CONSTRUCTION, not by a loss penalty, so the decision can
                      never be self-contradictory.

  SetArbiter       -- permutation-equivariant scoring over a whole local
                      neighbourhood, trained so that exactly one candidate
                      survives per sim track.  This is the DETR-style
                      formulation and it handles neighbourhoods of 3+ mutual
                      duplicates, which pairwise voting can resolve
                      inconsistently.

DATA REQUIREMENT
----------------
Both heads need a per-candidate QUALITY TARGET -- which of two duplicates is
the better reconstruction.  `t5_pMatched` (fraction of the candidate's hits
belonging to the matched sim track) is already loaded in the notebook's
`branches_list` and is exactly this quantity, but the pair-generation cells
currently discard it.  Carry it through alongside `sim_indices_per_event`.

Smoke test:  .venv/bin/python arbitration.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# pairwise
# ----------------------------------------------------------------------------
class PairwiseArbiter(nn.Module):
    """Antisymmetric keep/drop scoring for a duplicate pair.

    Writing the score as

        s(i, j) = g(e_i, e_j) - g(e_j, e_i)

    guarantees s(i,j) = -s(j,i) exactly, for any g and any weights, including
    at initialisation.  "Keep i" iff s(i,j) > 0.

    A naive head that concatenates [e_i, e_j] and regresses a scalar has no such
    guarantee: it can output "keep i" for (i,j) and also "keep j" for (j,i),
    and the resulting decision depends on the arbitrary order the pairs were
    enumerated in.  That is a real failure mode -- the notebook's
    `np.triu_indices` ordering is an implementation detail, not physics.
    """

    def __init__(self, emb_dim: int, extra_dim: int = 0, hidden: int = 64):
        super().__init__()
        self.extra_dim = extra_dim
        in_dim = 2 * emb_dim + 2 * extra_dim
        self.g = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def _g(self, a, b, xa=None, xb=None):
        parts = [a, b] if xa is None else [a, xa, b, xb]
        return self.g(torch.cat(parts, dim=1))

    def forward(self, e_i, e_j, x_i=None, x_j=None):
        """Returns s(i,j); positive means keep i.  Antisymmetric by construction."""
        return self._g(e_i, e_j, x_i, x_j) - self._g(e_j, e_i, x_j, x_i)

    @staticmethod
    def loss(score, keep_i, weight=None):
        """`keep_i` is 1.0 when candidate i is the better reconstruction.

        Logistic loss on the antisymmetric score. Because s is antisymmetric,
        this is equivalent to a Bradley-Terry ranking model over candidates and
        is invariant to how each pair was ordered.
        """
        target = keep_i.view(-1, 1)
        l = F.binary_cross_entropy_with_logits(score, target, reduction="none")
        if weight is not None:
            l = l * weight.view(-1, 1)
        return l.mean()


# ----------------------------------------------------------------------------
# set-level
# ----------------------------------------------------------------------------
class SetArbiter(nn.Module):
    """Permutation-equivariant keep-scoring over a local candidate neighbourhood.

    One self-attention block over the (variable-size, masked) set of candidates
    in a ΔR neighbourhood, then a per-candidate keep logit.  Equivariance means
    relabelling the candidates permutes the outputs identically -- so unlike the
    hard-coded collection priority, the decision does not depend on the order
    candidates happen to be stored in.

    Trained so that exactly one candidate survives per sim track, which is the
    one-to-one assignment property DETR uses to eliminate duplicates without an
    NMS post-step.
    """

    def __init__(self, emb_dim: int, extra_dim: int = 0, hidden: int = 64, n_heads: int = 4):
        super().__init__()
        d = hidden
        self.inp = nn.Linear(emb_dim + extra_dim, d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.out = nn.Linear(d, 1)

    def forward(self, emb, mask):
        """emb  (B, N, emb_dim[+extra]) ;  mask (B, N) True where a candidate is real.

        Returns per-candidate keep logits (B, N); entries where mask is False
        are meaningless and must be ignored by the caller.
        """
        h = self.inp(emb)
        pad = ~mask
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return self.out(h).squeeze(-1)

    @staticmethod
    def loss(logits, mask, group_id, quality, w_rank: float = 1.0, w_abs: float = 1.0):
        """One-survivor-per-sim-track objective.

        Two terms, and BOTH are needed:

        `rank` -- for each group of candidates sharing a sim track, a softmax
            over that group's logits trained toward the member with the highest
            `quality`.  This is the one-to-one assignment that makes duplicates
            structurally impossible, expressed for groups rather than DETR's
            global Hungarian matching: the group structure is already known here
            (candidates sharing a matched sim index), so the O(N^3) assignment
            solve DETR needs is unnecessary.

        `abs`  -- per-candidate BCE toward "am I my group's survivor?".
            A group softmax is invariant to adding a constant to every logit in
            the group, so it constrains only the RANKING and leaves the absolute
            scale free.  Without this term the model reaches near-zero rank loss
            while still putting several candidates above the keep threshold --
            measured at 1.39 kept per track in the smoke test.  This is the
            one-to-one / one-to-many hybrid that DDQ (arXiv:2303.12776) argues
            is necessary for end-to-end detectors.

        `group_id` is -1 for candidates matched to no sim track (fakes); those
        are trained toward a low keep score.
        """
        device = logits.device
        rank_total = torch.zeros((), device=device)
        n_groups = 0
        abs_logits, abs_targets = [], []

        for b in range(logits.shape[0]):
            valid = mask[b]
            if not valid.any():
                continue
            gids = group_id[b][valid]
            lg = logits[b][valid]
            q = quality[b][valid]

            for g in torch.unique(gids):
                sel = gids == g
                if g.item() < 0:
                    # unmatched candidates: never keep
                    abs_logits.append(lg[sel])
                    abs_targets.append(torch.zeros_like(lg[sel]))
                    continue

                tgt = torch.zeros_like(lg[sel])
                tgt[q[sel].argmax()] = 1.0
                abs_logits.append(lg[sel])
                abs_targets.append(tgt)

                if sel.sum() >= 2:
                    rank_total = rank_total + F.cross_entropy(
                        lg[sel].unsqueeze(0), q[sel].argmax().view(1)
                    )
                    n_groups += 1

        rank = rank_total / max(n_groups, 1)
        if abs_logits:
            a = F.binary_cross_entropy_with_logits(
                torch.cat(abs_logits), torch.cat(abs_targets)
            )
        else:
            a = torch.zeros((), device=device)
        return w_rank * rank + w_abs * a


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def arbitration_accuracy(score, keep_i):
    """Fraction of duplicate pairs where the better candidate is kept."""
    return float(((score.squeeze(-1) > 0).float() == keep_i.view(-1)).float().mean())


def antisymmetry_error(model, e_i, e_j, x_i=None, x_j=None):
    """Must be ~0 to machine precision.  A concatenation head will not pass."""
    with torch.no_grad():
        s_ij = model(e_i, e_j, x_i, x_j)
        s_ji = model(e_j, e_i, x_j, x_i)
        return float((s_ij + s_ji).abs().max())


def set_survivor_stats(logits, mask, group_id):
    """Duplicate rate and efficiency implied by 'keep the argmax of each group'.

    Reports what the arbitration achieves at the object level -- the quantity
    CMS MTV actually measures -- rather than a pairwise AUC.
    """
    kept_per_group, groups = [], 0
    for b in range(logits.shape[0]):
        valid = mask[b]
        if not valid.any():
            continue
        gids, lg = group_id[b][valid], logits[b][valid]
        for g in torch.unique(gids):
            if g.item() < 0:
                continue
            sel = gids == g
            kept_per_group.append(int((lg[sel] > 0).sum()))
            groups += 1
    if not groups:
        return {}
    k = np.array(kept_per_group)
    return {
        "n_sim_tracks": groups,
        "efficiency": float((k >= 1).mean()),       # sim track kept at all
        "duplicate_rate": float((k > 1).mean()),    # sim track kept more than once
        "mean_kept_per_track": float(k.mean()),
    }


# ----------------------------------------------------------------------------
# smoke test
# ----------------------------------------------------------------------------
def _smoke():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    emb_dim, extra = 16, 3

    print("=" * 70)
    print("PairwiseArbiter")
    print("=" * 70)
    n = 8000
    e_i = torch.randn(n, emb_dim)
    e_j = torch.randn(n, emb_dim)
    # quality is a hidden linear function of the extra features
    x_i, x_j = torch.randn(n, extra), torch.randn(n, extra)
    w = torch.randn(extra)
    keep_i = ((x_i @ w) > (x_j @ w)).float()

    m = PairwiseArbiter(emb_dim, extra)
    print(f"  antisymmetry error at init : {antisymmetry_error(m, e_i, e_j, x_i, x_j):.2e}")
    print(f"  accuracy at init           : {arbitration_accuracy(m(e_i, e_j, x_i, x_j), keep_i):.3f}")

    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for ep in range(1, 61):
        opt.zero_grad()
        s = m(e_i, e_j, x_i, x_j)
        l = PairwiseArbiter.loss(s, keep_i)
        l.backward(); opt.step()
        if ep % 20 == 0:
            print(f"  ep {ep:3d}  loss={l.item():.4f}  "
                  f"acc={arbitration_accuracy(s, keep_i):.3f}")
    print(f"  antisymmetry error trained : {antisymmetry_error(m, e_i, e_j, x_i, x_j):.2e}")

    # order-invariance: swapping the pair order must flip every decision
    with torch.no_grad():
        d_ij = (m(e_i, e_j, x_i, x_j) > 0)
        d_ji = (m(e_j, e_i, x_j, x_i) > 0)
    print(f"  decisions consistent under pair reordering: {bool((d_ij == ~d_ji).all())}")

    print("\n" + "=" * 70)
    print("SetArbiter")
    print("=" * 70)
    B, N = 256, 8
    emb = torch.randn(B, N, emb_dim + extra)
    n_real = rng.integers(3, N + 1, size=B)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b, k in enumerate(n_real):
        mask[b, :k] = True
    # each neighbourhood holds 2-3 sim tracks, some with several candidates
    gid = torch.full((B, N), -1, dtype=torch.long)
    for b, k in enumerate(n_real):
        gid[b, :k] = torch.tensor(rng.integers(0, 3, size=k))
    quality = emb[..., emb_dim:].sum(-1)  # hidden quality signal

    sm = SetArbiter(emb_dim + extra)
    opt = torch.optim.Adam(sm.parameters(), lr=3e-3)
    for ep in range(1, 121):
        opt.zero_grad()
        lg = sm(emb, mask)
        l = SetArbiter.loss(lg, mask, gid, quality)
        l.backward(); opt.step()
        if ep % 40 == 0:
            st = set_survivor_stats(lg.detach(), mask, gid)
            print(f"  ep {ep:3d}  loss={l.item():.4f}  "
                  f"eff={st['efficiency']:.3f}  dup={st['duplicate_rate']:.3f}  "
                  f"kept/track={st['mean_kept_per_track']:.2f}")

    # permutation equivariance
    perm = torch.randperm(N)
    with torch.no_grad():
        a = sm(emb, mask)
        b_ = sm(emb[:, perm], mask[:, perm])
    err = (a[:, perm] - b_)[mask[:, perm]].abs().max()
    print(f"  permutation-equivariance error : {float(err):.2e}")


if __name__ == "__main__":
    _smoke()
