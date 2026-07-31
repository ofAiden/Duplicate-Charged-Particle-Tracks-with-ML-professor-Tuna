"""
Corrected Euclidean / Mahalanobis / Cosine comparison for LST duplicate removal.

Fixes the four defects that make the comparison in updated_fixed-3.ipynb
uninterpretable:

  1. Each metric gets its OWN encoder pair, trained under its OWN objective.
     In the notebook only the Mahalanobis branch was optimised in cell 9, so the
     reported "Cosine AUC 0.7396 / 0.4832" came from randomly-initialised
     networks, and the "Euclidean" number was the Mahalanobis-trained embedding
     read out with a different distance.

  2. The Mahalanobis ridge is scale-free (eps * trace/d) instead of a fixed
     1e-3.  With the fixed value, 11 of 12 covariance eigenvalues sat below the
     ridge and 94.9% of ||V||_F^2 was pure rescaling -- i.e. Mahalanobis was
     Euclidean times a constant, which is why the two AUCs agreed to 4 d.p.
     See diagnose_metric_degeneracy.py.

  3. The margin is set relative to the scale the embedding actually occupies.
     The notebook used margin=1.0 while typical pair distances were ~0.09.

  4. The `0.01 * ||cov(e) - I||^2` term is replaced by a VICReg-style variance
     HINGE (arXiv:2105.04906).  The old term has no hinge and directly fights
     the Mahalanobis metric by trying to whiten the space that V is meant to
     model.  The hinge only pushes back when a dimension is about to die, so it
     prevents collapse without dictating the covariance.

Collapse is tracked with RankMe (arXiv:2210.02885) every epoch.

Usage
-----
    python metric_study.py --pairs pairs.npz --epochs 60 --out results/

`pairs.npz` must contain:
    X_left, X_right, y_t5, w_t5        (T5-T5 pairs;  y=0 duplicate, 1 not)
    X_pls,  X_t5cross, y_pls, w_pls    (pLS-T5 pairs)
Save it once from the notebook after the pair-generation cells.

Smoke test with synthetic data (no ROOT file needed):
    python metric_study.py --synthetic --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

T5_IN_DIM = 30
PLS_IN_DIM = 10


# ----------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------
def rankme(embeddings: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe effective rank (arXiv:2210.02885).

    p_k = sigma_k / ||sigma||_1 + eps ;  RankMe = exp(-sum p_k log p_k).
    Ranges from 1 (fully collapsed) to emb_dim (isotropic).
    """
    with torch.no_grad():
        sv = torch.linalg.svdvals(embeddings.float() - embeddings.float().mean(0, keepdim=True))
        p = sv / (sv.sum() + eps) + eps
        return float(torch.exp(-(p * p.log()).sum()))


def variance_explained(embeddings: torch.Tensor, k: int = 3) -> float:
    with torch.no_grad():
        sv = torch.linalg.svdvals(embeddings.float() - embeddings.float().mean(0, keepdim=True))
        var = sv ** 2
        return float(var[:k].sum() / var.sum())


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def make_encoder(input_dim: int, emb_dim: int, hidden: int = 128) -> nn.Sequential:
    """Wider encoder with BatchNorm, ported from updated_fixed_v2.ipynb.

    No terminal ReLU: it confines embeddings to the non-negative orthant, which
    both distorts the covariance geometry the Mahalanobis metric depends on and
    is a documented route to dimensional collapse.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, emb_dim),
    )


class MahalanobisMetric(nn.Module):
    """Mahalanobis distance with a periodically re-estimated precision matrix.

    The ridge is proportional to trace(cov)/d rather than a fixed constant, so
    the regulariser cannot swamp the covariance when the embedding lives at a
    small scale.
    """

    def __init__(self, emb_dim: int, ridge_rel: float = 1e-3):
        super().__init__()
        self.emb_dim = emb_dim
        self.ridge_rel = ridge_rel
        self.register_buffer("V", torch.eye(emb_dim))
        self.register_buffer("cov", torch.eye(emb_dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        dist_sq = ((diff @ self.V) * diff).sum(dim=1)
        return torch.sqrt(dist_sq.clamp(min=1e-12)).view(-1, 1)

    # A covariance estimated from ~d samples is almost pure noise. W-MSE
    # (arXiv:2007.06346) treats 2*d as the bare MINIMUM for a stable estimate;
    # RankMe (arXiv:2210.02885) uses 25600 samples for its spectrum. Estimating
    # a 12x12 covariance from 14 rows -- the old guard -- produces a V driven
    # entirely by sampling noise.
    MIN_COV_SAMPLES_PER_DIM = 100

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor) -> dict | None:
        need = max(self.emb_dim * self.MIN_COV_SAMPLES_PER_DIM, 1000)
        if embeddings.shape[0] < need:
            return None
        cov = torch.cov(embeddings.T.double())
        cov = (cov + cov.T) / 2
        scale = torch.diagonal(cov).mean().clamp(min=1e-30)
        ridge = self.ridge_rel * scale
        try:
            V = torch.linalg.inv(cov + ridge * torch.eye(self.emb_dim, device=cov.device, dtype=cov.dtype))
        except torch.linalg.LinAlgError:
            return None
        if not torch.isfinite(V).all():
            return None
        # Normalise V to unit mean diagonal so the distance scale stays
        # comparable to Euclidean and the margin means the same thing.
        V = V / torch.diagonal(V).mean().clamp(min=1e-30)
        self.V.copy_(V.to(self.V.dtype))
        self.cov.copy_(cov.to(self.cov.dtype))
        eig = torch.linalg.eigvalsh(cov)
        Vf = V.float()
        k = torch.diagonal(Vf).mean()
        return {
            "ridge": float(ridge),
            "n_eig_below_ridge": int((eig < ridge).sum()),
            "scalar_fraction": float((k ** 2 * self.emb_dim) / (Vf.norm() ** 2)),
            "cov_cond": float(eig.max() / eig.min().clamp(min=1e-30)),
        }


# ----------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------
class ContrastiveLoss(nn.Module):
    """Hadsell et al. 2006.  y=0 duplicate (pull), y=1 non-duplicate (push)."""

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, d, label, weight=None):
        loss = (1 - label) * d.pow(2) + label * (self.margin - d).clamp(min=0.0).pow(2)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


def vicreg_variance_hinge(emb: torch.Tensor, target_std: float,
                          two_sided: bool = True) -> torch.Tensor:
    """VICReg variance term (arXiv:2105.04906), eq. 1, made TWO-SIDED.

        v(Z) = (1/d) sum_j [ max(0, gamma - std_j) + max(0, std_j - c*gamma) ]

    VICReg's original term is a floor only: it stops a dimension dying but does
    nothing if the scale runs away upward.  That is fine for VICReg, whose loss
    is scale-free, but NOT here: the contrastive margin is a fixed length, so it
    is only meaningful relative to the scale the embedding occupies.

    Measured failure this fixes: with a floor-only hinge the Euclidean embedding
    grew to per-dimension std ~7 (pair distances ~9.5) against a margin of 3.46.
    Every pair then sat OUTSIDE the margin, `(margin - d).clamp(min=0)` was
    identically zero, the repulsive term contributed no gradient, and the
    embedding collapsed to RankMe 2.4/12 with cross-collection AUC 0.53. The
    hinge never fired (reg = 0.0000) because every per-dimension std exceeded the
    floor -- a floor cannot detect a scale that is too LARGE.

    Note this is a scale constraint, not a whitening constraint: it bounds each
    dimension's marginal std but leaves the off-diagonal covariance free, so the
    Mahalanobis metric still has something to model.
    """
    std = torch.sqrt(emb.var(dim=0) + 1e-8)
    loss = F.relu(target_std - std)
    if two_sided:
        # allow a factor of `c` of headroom before penalising growth
        loss = loss + F.relu(std - 3.0 * target_std)
    return loss.mean()


# ----------------------------------------------------------------------------
# distances
# ----------------------------------------------------------------------------
def euclidean(a, b):
    return torch.sqrt(((a - b) ** 2).sum(1, keepdim=True) + 1e-12)


def cosine(a, b, temperature: float):
    """Cosine distance with a temperature.

    At tau=1 an unscaled cosine distance is confined to [0, 2], so a margin of
    1.0 is only reachable at 90 degrees and the loss is badly conditioned.
    JetCLR uses tau=0.1-0.2, HEPTv2 uses 0.07.
    """
    return ((1.0 - F.cosine_similarity(a, b, dim=1, eps=1e-8)) / temperature).view(-1, 1)


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
@dataclass
class Config:
    emb_dim: int = 16
    hidden: int = 128
    batch_size: int = 1024
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_norm: float = 5.0

    # margin is set from the data scale unless overridden
    margin: float | None = None
    target_std: float = 0.5       # VICReg gamma; also fixes the embedding scale
    vicreg_weight: float = 1.0
    cosine_temperature: float = 0.15

    cov_update_interval: int = 2
    max_cov_samples: int = 50000
    ridge_rel: float = 1e-3
    seed: int = 42
    metrics: tuple = ("euclidean", "mahalanobis", "cosine")


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
class MetricBranch:
    """One metric = one independent pair of encoders + its own optimiser."""

    def __init__(self, name: str, cfg: Config, device):
        self.name = name
        self.cfg = cfg
        self.device = device
        self.enc_t5 = make_encoder(T5_IN_DIM, cfg.emb_dim, cfg.hidden).to(device)
        self.enc_pls = make_encoder(PLS_IN_DIM, cfg.emb_dim, cfg.hidden).to(device)
        params = list(self.enc_t5.parameters()) + list(self.enc_pls.parameters())

        self.metric = None
        if name == "mahalanobis":
            # Separate precision matrices for the T5-T5 and pLS-T5 populations:
            # they are different object types with different covariance, so one
            # shared V would be misspecified.
            self.metric = MahalanobisMetric(cfg.emb_dim, cfg.ridge_rel).to(device)
            self.metric_cross = MahalanobisMetric(cfg.emb_dim, cfg.ridge_rel).to(device)

        self.opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=8
        )
        # margin defaults to ~2x the scale the variance hinge pins the
        # embedding to, so pairs actually populate both sides of the margin
        margin = cfg.margin if cfg.margin is not None else 2.0 * cfg.target_std * math.sqrt(cfg.emb_dim)
        self.margin = margin
        self.criterion = ContrastiveLoss(margin)
        self.history: list[dict] = []

    def distance(self, a, b, cross: bool):
        if self.name == "euclidean":
            return euclidean(a, b)
        if self.name == "cosine":
            return cosine(a, b, self.cfg.cosine_temperature)
        m = self.metric_cross if cross else self.metric
        return m(a, b)

    def train(self):
        self.enc_t5.train(); self.enc_pls.train()

    def eval(self):
        self.enc_t5.eval(); self.enc_pls.eval()

    def state_dict(self):
        s = {"enc_t5": self.enc_t5.state_dict(), "enc_pls": self.enc_pls.state_dict(),
             "margin": self.margin}
        if self.metric is not None:
            s["metric"] = self.metric.state_dict()
            s["metric_cross"] = self.metric_cross.state_dict()
        return s


def run_epoch(branch: MetricBranch, t5_loader, pls_loader, cfg: Config, train: bool):
    branch.train() if train else branch.eval()
    tot, tot_t5, tot_pls, tot_reg, nb = 0.0, 0.0, 0.0, 0.0, 0
    tot_active = 0.0
    emb_buf, emb_cross_buf, n_buf = [], [], 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (l, r, y0, w0), (p5, t5f, y1, w1) in zip(t5_loader, pls_loader):
            dev = branch.device
            l, r, y0, w0 = l.to(dev), r.to(dev), y0.to(dev), w0.to(dev)
            p5, t5f, y1, w1 = p5.to(dev), t5f.to(dev), y1.to(dev), w1.to(dev)

            e_l, e_r = branch.enc_t5(l), branch.enc_t5(r)
            e_p, e_t = branch.enc_pls(p5), branch.enc_t5(t5f)

            d_t5 = branch.distance(e_l, e_r, False)
            d_pls = branch.distance(e_p, e_t, True)
            loss_t5 = branch.criterion(d_t5, y0, w0)
            loss_pls = branch.criterion(d_pls, y1, w1)

            # MARGIN HEALTH: fraction of non-duplicate pairs still inside the
            # margin, i.e. still producing a repulsive gradient. If this hits 0
            # the repulsive term is dead and the embedding will collapse; if it
            # stays at 1 the margin is so wide it never saturates. Either way the
            # run is silently broken, which is exactly what happened before this
            # check existed.
            with torch.no_grad():
                nd = y0.squeeze(-1) == 1
                active = float((d_t5.squeeze(-1)[nd] < branch.margin).float().mean()) \
                    if nd.any() else float("nan")

            # cosine is scale-invariant, so a variance hinge on the raw
            # embedding would be meaningless there
            if branch.name == "cosine":
                reg = torch.zeros((), device=dev)
            else:
                # VICReg computes the variance term on EACH BRANCH INDEPENDENTLY
                # (arXiv:2105.04906 sec. 4.1). That is precisely why it tolerates
                # branches with different architectures and unshared weights --
                # the property this project needs, since the T5 and pLS towers
                # differ. Pooling two towers into one variance estimate can
                # satisfy the hinge while one tower is individually collapsed,
                # so each of the four embedding streams gets its own hinge.
                reg = cfg.vicreg_weight * 0.25 * (
                    vicreg_variance_hinge(e_l, cfg.target_std)
                    + vicreg_variance_hinge(e_r, cfg.target_std)
                    + vicreg_variance_hinge(e_p, cfg.target_std)
                    + vicreg_variance_hinge(e_t, cfg.target_std)
                )

            loss = loss_t5 + loss_pls + reg

            if train:
                branch.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(branch.enc_t5.parameters()) + list(branch.enc_pls.parameters()),
                    cfg.clip_norm,
                )
                branch.opt.step()

            tot += loss.item(); tot_t5 += loss_t5.item()
            tot_pls += loss_pls.item(); tot_reg += float(reg.detach()); nb += 1
            if np.isfinite(active):
                tot_active += active

            if n_buf < cfg.max_cov_samples:
                sim0, sim1 = (y0.squeeze(-1) == 0), (y1.squeeze(-1) == 0)
                if sim0.any():
                    emb_buf.append(torch.cat([e_l[sim0], e_r[sim0]]).detach())
                if sim1.any():
                    emb_cross_buf.append(torch.cat([e_p[sim1], e_t[sim1]]).detach())
                n_buf += int(sim0.sum()) * 2

    nb = max(nb, 1)
    stats = {"loss": tot / nb, "loss_t5": tot_t5 / nb, "loss_pls": tot_pls / nb,
             "reg": tot_reg / nb, "margin_active": tot_active / nb}
    embs = torch.cat(emb_buf) if emb_buf else None
    embs_cross = torch.cat(emb_cross_buf) if emb_cross_buf else None
    return stats, embs, embs_cross


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect_distances(branch: MetricBranch, loader, cross: bool):
    branch.eval()
    ds, ys = [], []
    for a, b, y, _w in loader:
        a, b = a.to(branch.device), b.to(branch.device)
        enc_a = branch.enc_pls if cross else branch.enc_t5
        d = branch.distance(enc_a(a), branch.enc_t5(b), cross)
        ds.append(d.cpu().numpy().ravel()); ys.append(y.numpy().ravel())
    return np.concatenate(ds), np.concatenate(ys)


def apply_blocking_ceiling(recall_in_window: float, ceiling: float) -> float:
    """Convert a recall measured on windowed pairs into end-to-end recall.

    The dR^2 < 0.02 window is a blocking step: true duplicate pairs whose
    members fall outside it are never presented to the model and can never be
    recovered.  A recall of 0.95 measured on windowed pairs, behind a window
    that only contains 80% of true duplicates, is an end-to-end recall of 0.76.

    Reporting the windowed number alone overstates performance, and the
    entity-resolution literature treats blocking recall as mandatory to report
    for exactly this reason.  build_pairs.py measures the ceiling; this applies
    it.
    """
    if not np.isfinite(ceiling):
        return float("nan")
    return recall_in_window * ceiling


def duplicate_metrics_at_prior(d, y, prior: float, target_eff: float = 0.99, rng=None):
    """Re-weight the artificially balanced test set to a realistic duplicate
    prior, then report the operating point that keeps `target_eff` of genuine
    (non-duplicate) tracks.

    The notebook's 50/50 pair sampling makes AUC read far better than it will in
    deployment, where only a few percent of nearby pairs are true duplicates.
    y=0 duplicate, y=1 non-duplicate.
    """
    rng = rng or np.random.default_rng(0)
    dup, non = d[y == 0], d[y == 1]
    n_non = len(non)
    n_dup = int(round(prior / (1 - prior) * n_non))
    if n_dup == 0 or len(dup) == 0:
        return None
    dup_s = rng.choice(dup, size=n_dup, replace=n_dup > len(dup))

    # threshold below which a pair is declared duplicate; choose it so that
    # only (1 - target_eff) of genuine tracks are wrongly removed
    thr = np.quantile(non, 1 - target_eff)
    return {
        "prior": prior,
        "threshold": float(thr),
        "target_efficiency": target_eff,
        "duplicate_recall": float((dup_s < thr).mean()),
        "false_removal_rate": float((non < thr).mean()),
        "auc_balanced": float(roc_auc_score(y, d)),
    }


def conformal_threshold(d_non_dup_calib: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold with a distribution-free guarantee.

    Choosing thr as the floor(alpha*(n+1))-th smallest distance among genuine
    NON-duplicate pairs guarantees, over the draw of the calibration set, that
    the probability a future genuine pair is wrongly declared a duplicate is at
    most alpha (arXiv:2107.07511).

    Assumes exchangeability between calibration and deployment -- it does NOT
    survive a pileup or detector-condition shift, which must be stated.
    """
    n = len(d_non_dup_calib)
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        return float(-np.inf)
    return float(np.sort(d_non_dup_calib)[k - 1])


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def make_loaders(arrays: dict, cfg: Config, device):
    def split(*xs, frac=0.2, seed=42):
        n = len(xs[0])
        idx = np.random.default_rng(seed).permutation(n)
        cut = int(n * (1 - frac))
        return [x[idx[:cut]] for x in xs], [x[idx[cut:]] for x in xs]

    tr_t5, te_t5 = split(arrays["X_left"], arrays["X_right"], arrays["y_t5"], arrays["w_t5"])
    tr_pl, te_pl = split(arrays["X_pls"], arrays["X_t5cross"], arrays["y_pls"], arrays["w_pls"])

    def ds(parts):
        a, b, y, w = parts
        return TensorDataset(
            torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1),
            torch.tensor(w, dtype=torch.float32).view(-1, 1),
        )

    pin = device.type == "cuda"
    mk = lambda d, sh: DataLoader(d, cfg.batch_size, shuffle=sh, drop_last=sh, pin_memory=pin)
    return (mk(ds(tr_t5), True), mk(ds(te_t5), False),
            mk(ds(tr_pl), True), mk(ds(te_pl), False))


def synthetic(n=20000, seed=0):
    """Smoke-test data with genuine pair structure, so the harness can be
    validated without the 20-minute pair-generation step."""
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal((n, 6))
    y_t5 = (rng.random(n) < 0.5).astype(np.float32)
    W5 = rng.standard_normal((6, T5_IN_DIM))
    left = lat @ W5 + 0.1 * rng.standard_normal((n, T5_IN_DIM))
    partner = np.where(y_t5[:, None] == 0, lat, rng.standard_normal((n, 6)))
    right = partner @ W5 + 0.1 * rng.standard_normal((n, T5_IN_DIM))

    y_p = (rng.random(n) < 0.5).astype(np.float32)
    Wp = rng.standard_normal((6, PLS_IN_DIM))
    lat_p = rng.standard_normal((n, 6))
    pls = lat_p @ Wp + 0.1 * rng.standard_normal((n, PLS_IN_DIM))
    partner_p = np.where(y_p[:, None] == 0, lat_p, rng.standard_normal((n, 6)))
    t5c = partner_p @ W5 + 0.1 * rng.standard_normal((n, T5_IN_DIM))

    f32 = lambda a: a.astype(np.float32)
    return {"X_left": f32(left), "X_right": f32(right), "y_t5": y_t5,
            "w_t5": np.ones(n, np.float32), "X_pls": f32(pls), "X_t5cross": f32(t5c),
            "y_pls": y_p, "w_pls": np.ones(n, np.float32)}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=None, help="npz produced by the notebook")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--emb-dim", type=int, default=Config.emb_dim)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    cfg = Config(epochs=args.epochs, emb_dim=args.emb_dim)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ceilings = {"t5": float("nan"), "pls": float("nan")}
    if args.synthetic:
        arrays = synthetic()
    elif args.pairs:
        arrays = {k: v for k, v in np.load(args.pairs).items()}
        for side, key in (("t5", "blocking_ceiling_t5"), ("pls", "blocking_ceiling_pls")):
            if key in arrays:
                ceilings[side] = float(arrays.pop(key))
        if np.isfinite(ceilings["t5"]):
            print(f"blocking recall ceiling: T5-T5 {ceilings['t5']:.4f}  "
                  f"pLS-T5 {ceilings['pls']:.4f}")
        else:
            print("blocking recall ceiling: NOT MEASURED -- regenerate pairs with "
                  "build_pairs.py.\n  Every recall below is conditional on the "
                  "dR^2 window and overstates end-to-end performance.")
    else:
        raise SystemExit("pass --pairs pairs.npz or --synthetic")

    tr_t5, te_t5, tr_pl, te_pl = make_loaders(arrays, cfg, device)
    print(f"device={device}  emb_dim={cfg.emb_dim}  epochs={cfg.epochs}")
    print(f"train batches: T5 {len(tr_t5)}  pLS {len(tr_pl)}")

    results = {}
    for name in cfg.metrics:
        print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
        br = MetricBranch(name, cfg, device)
        print(f"margin={br.margin:.3f}")

        for ep in range(1, cfg.epochs + 1):
            stats, embs, embs_cross = run_epoch(br, tr_t5, tr_pl, cfg, train=True)
            cov_info = None
            if name == "mahalanobis" and ep % cfg.cov_update_interval == 0:
                if embs is not None:
                    cov_info = br.metric.update(embs[: cfg.max_cov_samples])
                if embs_cross is not None:
                    br.metric_cross.update(embs_cross[: cfg.max_cov_samples])
            br.sched.step(stats["loss"])

            rec = {"epoch": ep, **stats}
            if embs is not None:
                rec["rankme"] = rankme(embs)
                rec["var_top3"] = variance_explained(embs, 3)
            if cov_info:
                rec.update({f"cov_{k}": v for k, v in cov_info.items()})
            br.history.append(rec)

            if ep % 5 == 0 or ep == 1:
                extra = ""
                if "rankme" in rec:
                    extra = f"  RankMe={rec['rankme']:.2f}/{cfg.emb_dim}  var@3={rec['var_top3']:.1%}"
                if cov_info:
                    extra += (f"  V_scalar={cov_info['scalar_fraction']:.1%}"
                              f"  eig<ridge={cov_info['n_eig_below_ridge']}/{cfg.emb_dim}")
                warn = ""
                a = stats["margin_active"]
                if a < 0.02:
                    warn = "  ** MARGIN TOO SMALL: repulsion dead **"
                elif a > 0.98:
                    warn = "  ** MARGIN TOO WIDE: never saturates **"
                print(f"  ep {ep:3d}  loss={stats['loss']:.4f}  "
                      f"t5={stats['loss_t5']:.4f}  pls={stats['loss_pls']:.4f}"
                      f"  reg={stats['reg']:.4f}  active={a:.2f}{extra}{warn}")

        d_t5, y_t5 = collect_distances(br, te_t5, cross=False)
        d_pl, y_pl = collect_distances(br, te_pl, cross=True)

        res = {
            "margin": br.margin,
            "auc_t5_balanced": float(roc_auc_score(y_t5, d_t5)),
            "auc_pls_balanced": float(roc_auc_score(y_pl, d_pl)),
            "history": br.history,
        }
        # realistic-prior operating points; LST's residual duplicate rate is a
        # few percent, not the 50% the balanced pair sampling implies
        for prior in (0.05, 0.02):
            for side, dd, yy in (("t5", d_t5, y_t5), ("pls", d_pl, y_pl)):
                op = duplicate_metrics_at_prior(dd, yy, prior)
                if op is not None:
                    op["blocking_ceiling"] = ceilings[side]
                    op["duplicate_recall_end_to_end"] = apply_blocking_ceiling(
                        op["duplicate_recall"], ceilings[side]
                    )
                res[f"{side}_at_prior_{prior}"] = op

        # conformal threshold with a guaranteed false-removal rate
        non_dup = d_t5[y_t5 == 1]
        half = len(non_dup) // 2
        for alpha in (0.005, 0.01):
            thr = conformal_threshold(non_dup[:half], alpha)
            res[f"conformal_alpha_{alpha}"] = {
                "threshold": thr,
                "empirical_false_removal_holdout": float((non_dup[half:] < thr).mean()),
                "duplicate_recall": float((d_t5[y_t5 == 0] < thr).mean()),
            }

        results[name] = res
        torch.save(br.state_dict(), out / f"{name}.pt")
        print(f"\n  AUC (balanced)  T5-T5 {res['auc_t5_balanced']:.4f}   "
              f"pLS-T5 {res['auc_pls_balanced']:.4f}")
        op = res["t5_at_prior_0.05"]
        if op:
            print(f"  @5% prior, 99% track efficiency: duplicate recall "
                  f"{op['duplicate_recall']:.3f}")
        cf = res["conformal_alpha_0.005"]
        print(f"  conformal alpha=0.005: thr={cf['threshold']:.4f}  "
              f"holdout false-removal={cf['empirical_false_removal_holdout']:.4f}  "
              f"dup recall={cf['duplicate_recall']:.3f}")

    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 70}\nSUMMARY (all three trained independently on their own objective)\n{'=' * 70}")
    print(f"{'metric':<14}{'AUC T5':>10}{'AUC pLS':>10}{'RankMe':>10}{'var@3':>9}")
    for k, v in results.items():
        last = v["history"][-1]
        print(f"{k:<14}{v['auc_t5_balanced']:>10.4f}{v['auc_pls_balanced']:>10.4f}"
              f"{last.get('rankme', float('nan')):>10.2f}{last.get('var_top3', float('nan')):>9.1%}")
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
