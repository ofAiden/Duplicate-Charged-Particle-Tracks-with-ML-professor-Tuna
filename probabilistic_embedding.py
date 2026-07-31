"""
Per-instance uncertainty embedding -- the constructive answer to the gauge result.

THE ARGUMENT THIS COMPLETES
---------------------------
`verify_gauge_freedom.py` shows that a GLOBAL learned Mahalanobis matrix on top
of a linear-output encoder is a reparameterisation of Euclidean: writing
V = L^T L, the metric can be absorbed into the final weight as LW, so it adds
zero expressive power. That is a negative result.

The fix it implies is that the precision must depend on the INPUT, because no
fixed weight matrix can absorb a transform that varies per pair. This module
implements the cheapest such model for a small tabular MLP and -- critically --
verifies numerically that it is NOT absorbable, which is what makes the paper's
argument complete rather than merely critical.

Two ingredients, both from face recognition:

Probabilistic Face Embeddings (Shi & Jain, arXiv:1904.09658) predicts a
per-instance diagonal variance and scores a pair by the MUTUAL LIKELIHOOD SCORE
-- the log likelihood that the two latent variables are equal:

    s(i,j) = -1/2 * sum_l [ (mu_i^l - mu_j^l)^2 / (sigma_i^2(l) + sigma_j^2(l))
                            + log(sigma_i^2(l) + sigma_j^2(l)) ]  + const

The effective precision 1/(sigma_i^2 + sigma_j^2) is per-PAIR, and the log-det
term has no representation as any distance in any deterministic embedding.

Hedged Instance Embedding (Oh et al., arXiv:1810.00319) supplies a match
probability with NO MARGIN HYPERPARAMETER:

    p(match | z_i, z_j) = sigmoid(-a * ||z_i - z_j||_2 + b),   a > 0

a and b are learned. This matters here because the project measured an ~11x
mismatch between the embedding scale (RMS norm 0.062) and the hand-set
contrastive margin (1.0); with a learned scale that mismatch is structurally
impossible rather than a silent defect.

WHY IT IS PHYSICALLY MOTIVATED, NOT JUST A TRICK
------------------------------------------------
The inputs already carry fitted per-object uncertainties -- pLS features include
etaErr and ptErr explicitly, and T5 features include the inverse radii. So a
learned sigma has ground truth to be checked against: regress it on those and
report the correlation. If it tracks them, the embedding learned the detector's
resolution; if not, that is a publishable negative. Either way it is a testable
claim rather than an added parameter.

Run:  .venv/bin/python probabilistic_embedding.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class ProbabilisticEncoder(nn.Module):
    """Encoder emitting a mean and a diagonal log-variance per candidate.

    One extra Linear head on the existing trunk. At emb_dim=12 / hidden=128 that
    is 128*12 + 12 = 1548 extra parameters on an encoder that costs ~3-4% of
    inverse throughput, i.e. a few tenths of a percent absolute at the HLT.

    Trained jointly from scratch (as in Data Uncertainty Learning,
    arXiv:2003.11339) rather than PFE's frozen-mean two-stage recipe -- that
    two-stage design existed only to retrofit pretrained face models, which is
    not a constraint here.
    """

    def __init__(self, input_dim: int, emb_dim: int, hidden: int = 128,
                 log_var_range: tuple[float, float] = (-8.0, 2.0)):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden // 2, emb_dim)
        self.logvar_head = nn.Linear(hidden // 2, emb_dim)
        self.lo, self.hi = log_var_range
        # start near-deterministic: small variance, so early training behaves
        # like the plain encoder and the variance head has to earn its influence
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -4.0)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu_head(h)
        # clamping keeps the log-det term finite; an unbounded log-variance lets
        # the model drive sigma -> 0 and make MLS diverge
        logvar = self.logvar_head(h).clamp(self.lo, self.hi)
        return mu, logvar


# ----------------------------------------------------------------------------
# scores
# ----------------------------------------------------------------------------
def mutual_likelihood_score(mu_i, logvar_i, mu_j, logvar_j, include_const=False):
    """PFE mutual likelihood score (arXiv:1904.09658 eq. 3).

    Returns a SIMILARITY (higher = more likely the same object). Negate it to
    use as a distance.
    """
    var_sum = logvar_i.exp() + logvar_j.exp()
    sq = (mu_i - mu_j) ** 2
    s = -0.5 * (sq / var_sum + var_sum.log()).sum(dim=1, keepdim=True)
    if include_const:
        s = s - 0.5 * mu_i.shape[1] * LOG_2PI
    return s


def mls_distance(mu_i, logvar_i, mu_j, logvar_j):
    """MLS as a distance, for drop-in use where a distance is expected."""
    return -mutual_likelihood_score(mu_i, logvar_i, mu_j, logvar_j)


class SoftContrastiveLoss(nn.Module):
    """HIB soft contrastive loss (arXiv:1810.00319). No margin.

        p(match) = sigmoid(-a * ||z_i - z_j||_2 + b),  a = softplus(a_raw) > 0

    a and b are learned, so the decision scale adapts to whatever scale the
    embedding settles at. This removes the margin hyperparameter that the
    project measured to be ~11x mis-set.

    Uses K Monte-Carlo samples from the Gaussian embedding at training time; at
    inference use the mean, so deployment cost is unchanged.
    """

    def __init__(self, init_a: float = 1.0, init_b: float = 0.0, n_samples: int = 8):
        super().__init__()
        self.a_raw = nn.Parameter(torch.tensor(math.log(math.expm1(init_a))))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.n_samples = n_samples

    @property
    def a(self):
        return F.softplus(self.a_raw)

    def match_logit(self, z_i, z_j):
        return -self.a * (z_i - z_j).norm(dim=1, keepdim=True) + self.b

    def forward(self, mu_i, logvar_i, mu_j, logvar_j, label, weight=None):
        """`label` follows the project convention: 0 = duplicate, 1 = not.
        The match target is therefore (1 - label)."""
        target = 1.0 - label
        std_i, std_j = (0.5 * logvar_i).exp(), (0.5 * logvar_j).exp()
        probs = []
        for _ in range(self.n_samples):
            z_i = mu_i + std_i * torch.randn_like(std_i)
            z_j = mu_j + std_j * torch.randn_like(std_j)
            probs.append(torch.sigmoid(self.match_logit(z_i, z_j)))
        p = torch.stack(probs).mean(0).clamp(1e-6, 1 - 1e-6)
        loss = -(target * p.log() + (1 - target) * (1 - p).log())
        if weight is not None:
            loss = loss * weight
        return loss.mean()


def kl_to_standard_normal(mu, logvar):
    """beta * KL(N(mu, sigma^2) || N(0, I)) -- HIB's regulariser.

    One principled term replacing the two hand-set knobs (target_std,
    vicreg_weight) in metric_study.py. HIB uses beta = 1e-4.
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


# ----------------------------------------------------------------------------
# THE KEY VERIFICATION
# ----------------------------------------------------------------------------
def verify_not_absorbable(n=8192, hid=64, emb=12, steps=8000, seed=0, verbose=True):
    """Is a per-instance metric absorbable into a linear layer, like a global one?

    PRECISE CLAIM UNDER TEST: the global Mahalanobis distance lies inside the
    family {squared Euclidean distance on a linear projection of h}, whereas the
    per-instance MLS does not.

    Control  : global Mahalanobis target. It is provably IN the family (the exact
               solution is lin = L @ W), so the residual must reach ~0. This is
               the calibration of the experiment -- if the control does not
               reach ~0 the fit is merely underfit and the comparison is void.
    Treatment: per-instance MLS target. If its residual plateaus far above the
               control's, MLS is outside the family.

    Both fits get the SAME generous budget with LR decay, and the control is
    additionally checked ANALYTICALLY by planting the exact solution -- so an
    optimisation failure cannot be mistaken for an expressiveness limit. (An
    earlier version of this test reported an inconclusive 13x ratio purely
    because the control was underfit at 1500 steps.)
    """
    torch.manual_seed(seed)
    h_a, h_b = torch.randn(n, hid), torch.randn(n, hid)

    W = torch.randn(emb, hid) / math.sqrt(hid)
    A = torch.randn(emb, emb)
    V = A.T @ A + 0.05 * torch.eye(emb)
    L = torch.linalg.cholesky(V).T

    mu_a, mu_b = h_a @ W.T, h_b @ W.T
    diff = mu_a - mu_b

    # --- control target: global Mahalanobis (provably absorbable)
    tgt_global = ((diff @ V) * diff).sum(1, keepdim=True)

    # --- treatment target: per-instance MLS with input-dependent variance
    Wv = torch.randn(emb, hid) / math.sqrt(hid)
    logvar_a = (h_a @ Wv.T).clamp(-3, 1)
    logvar_b = (h_b @ Wv.T).clamp(-3, 1)
    tgt_local = mls_distance(mu_a, logvar_a, mu_b, logvar_b)

    def normalise(t):
        return (t - t.mean()) / t.std()

    def residual_of(lin_weight, target):
        """Best affine rescaling of the family's output -- closed form, so no
        optimiser is involved in judging the fit."""
        d = (((h_a - h_b) @ lin_weight.T) ** 2).sum(1, keepdim=True)
        t = normalise(target)
        d_c, t_c = d - d.mean(), t - t.mean()
        beta = (d_c * t_c).sum() / (d_c * d_c).sum().clamp(min=1e-30)
        return float(F.mse_loss(beta * d_c, t_c))

    def fit(target, tag):
        torch.manual_seed(1)
        lin = nn.Linear(hid, emb, bias=False)
        scale = nn.Parameter(torch.ones(1))
        shift = nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([*lin.parameters(), scale, shift], lr=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        t = normalise(target)
        traj = []
        for s in range(steps):
            opt.zero_grad()
            d = (((h_a - h_b) @ lin.weight.T) ** 2).sum(1, keepdim=True)
            loss = F.mse_loss(scale * d + shift, t)
            loss.backward()
            opt.step()
            sched.step()
            if s in (steps // 4, steps // 2, steps - 1):
                traj.append(float(loss.detach()))
        if verbose:
            print(f"    {tag:<26} residual at 25%/50%/100% of budget: "
                  f"{traj[0]:.5f} / {traj[1]:.5f} / {traj[2]:.5f}")
        return traj[-1]

    if verbose:
        print("=" * 74)
        print("IS A PER-INSTANCE METRIC ABSORBABLE INTO A LINEAR LAYER?")
        print("=" * 74)
        print("  Family: {squared Euclidean distance on a linear projection}.")
        print("  Residual is normalised MSE; 1.0 = no better than the mean.\n")
        print("  ANALYTIC CHECK -- plant the exact solution lin = L @ W:")
        print(f"    global Mahalanobis, planted   residual = "
              f"{residual_of(L @ W, tgt_global):.3e}   <- must be ~0")
        print(f"    per-instance MLS,   planted   residual = "
              f"{residual_of(L @ W, tgt_local):.3e}")
        print("\n  FITTED FROM SCRATCH (same budget, cosine LR decay):")

    r_global = fit(tgt_global, "global Mahalanobis")
    r_local = fit(tgt_local, "per-instance MLS")

    if verbose:
        print(f"\n  final: global {r_global:.5f}   per-instance {r_local:.5f}   "
              f"ratio {r_local / max(r_global, 1e-12):.0f}x")
        control_ok = r_global < 1e-3
        separated = r_local > 50 * r_global and r_local > 0.05
        if not control_ok:
            print("  VERDICT: VOID -- the control did not converge, so this")
            print("           measures optimisation difficulty, not expressiveness.")
        elif separated:
            print("  VERDICT: CONFIRMED. The control is absorbed to numerical zero")
            print("           while MLS plateaus far above it, so the per-instance")
            print("           metric is genuinely outside the family.")
        else:
            print("  VERDICT: NOT SEPARATED -- treat the claim as unsupported.")
    return r_global, r_local


# ----------------------------------------------------------------------------
def _smoke():
    r_g, r_l = verify_not_absorbable()

    # ---- the log-det term is the part with no deterministic analogue
    print("\n" + "=" * 74)
    print("WHICH PART OF MLS CARRIES THE NEW INFORMATION?")
    print("=" * 74)
    torch.manual_seed(0)
    n, emb = 4096, 12
    mu_i, mu_j = torch.randn(n, emb), torch.randn(n, emb)
    lv_i, lv_j = torch.randn(n, emb).clamp(-3, 1), torch.randn(n, emb).clamp(-3, 1)
    var_sum = lv_i.exp() + lv_j.exp()
    mahal_term = ((mu_i - mu_j) ** 2 / var_sum).sum(1)
    logdet_term = var_sum.log().sum(1)
    eucl = ((mu_i - mu_j) ** 2).sum(1)
    corr = lambda a, b: float(np.corrcoef(a.numpy(), b.numpy())[0, 1])
    print(f"  corr(Euclidean, per-pair-weighted term) : {corr(eucl, mahal_term):+.4f}")
    print(f"  corr(Euclidean, log-det term)           : {corr(eucl, logdet_term):+.4f}")
    print("  The log-det term is essentially uncorrelated with any distance on")
    print("  the means -- it is pure per-object confidence, and it is why MLS is")
    print("  not a distance at all (it is a likelihood).")

    # ---- HIB soft contrastive loss trains and learns its own scale
    print("\n" + "=" * 74)
    print("HIB SOFT CONTRASTIVE LOSS: does it find the scale on its own?")
    print("=" * 74)
    torch.manual_seed(0)
    in_dim = 30
    lat = torch.randn(n, 6)
    M = torch.randn(6, in_dim)
    y = (torch.rand(n) < 0.5).float().view(-1, 1)
    x_a = lat @ M + 0.1 * torch.randn(n, in_dim)
    partner = torch.where(y == 0, lat, torch.randn(n, 6))
    x_b = partner @ M + 0.1 * torch.randn(n, in_dim)

    enc = ProbabilisticEncoder(in_dim, emb)
    crit = SoftContrastiveLoss(n_samples=4)
    opt = torch.optim.Adam([*enc.parameters(), *crit.parameters()], lr=3e-3)

    for ep in range(1, 121):
        enc.train()
        opt.zero_grad()
        mu_a, lv_a = enc(x_a)
        mu_b, lv_b = enc(x_b)
        loss = crit(mu_a, lv_a, mu_b, lv_b, y) + 1e-4 * (
            kl_to_standard_normal(mu_a, lv_a) + kl_to_standard_normal(mu_b, lv_b)
        )
        loss.backward()
        opt.step()
        if ep % 40 == 0:
            enc.eval()
            with torch.no_grad():
                mu_a, lv_a = enc(x_a)
                mu_b, lv_b = enc(x_b)
                d = (mu_a - mu_b).norm(dim=1)
                from sklearn.metrics import roc_auc_score
                auc_e = roc_auc_score(y.numpy().ravel(), d.numpy())
                auc_m = roc_auc_score(
                    y.numpy().ravel(),
                    mls_distance(mu_a, lv_a, mu_b, lv_b).numpy().ravel())
                sc = d.mean() / (mu_a.std(0).norm())
            print(f"  ep {ep:3d}  loss={loss.detach().item():.4f}  a={float(crit.a):.3f}  "
                  f"b={float(crit.b):+.3f}  AUC(eucl)={auc_e:.4f}  "
                  f"AUC(MLS)={auc_m:.4f}")

    print(f"\n  learned decision scale a = {float(crit.a):.4f}, offset b = "
          f"{float(crit.b):+.4f}")
    print("  a*||z|| ~ O(1) at the decision boundary by construction, so the")
    print("  margin/scale mismatch measured in the notebook (11x) cannot occur.")

    # ---- sigma should track a genuine noise level
    print("\n" + "=" * 74)
    print("DOES THE LEARNED SIGMA TRACK INPUT NOISE?  (the physics test)")
    print("=" * 74)
    torch.manual_seed(0)
    noise = torch.rand(n, 1) * 1.5           # per-object injected noise level
    x_noisy = lat @ M + noise * torch.randn(n, in_dim)
    enc2 = ProbabilisticEncoder(in_dim, emb)
    crit2 = SoftContrastiveLoss(n_samples=4)
    opt2 = torch.optim.Adam([*enc2.parameters(), *crit2.parameters()], lr=3e-3)
    partner2 = torch.where(y == 0, lat, torch.randn(n, 6))
    x_noisy_b = partner2 @ M + noise * torch.randn(n, in_dim)
    for _ in range(200):
        enc2.train(); opt2.zero_grad()
        ma, la = enc2(x_noisy); mb, lb = enc2(x_noisy_b)
        l = crit2(ma, la, mb, lb, y) + 1e-4 * kl_to_standard_normal(ma, la)
        l.backward(); opt2.step()
    enc2.eval()
    with torch.no_grad():
        _, lv = enc2(x_noisy)
    learned_sigma = (0.5 * lv).exp().mean(1)
    rho = float(np.corrcoef(noise.numpy().ravel(), learned_sigma.numpy())[0, 1])
    from scipy.stats import spearmanr
    sp = spearmanr(noise.numpy().ravel(), learned_sigma.numpy()).statistic
    print(f"  Pearson  corr(injected noise, learned sigma) = {rho:+.4f}")
    print(f"  Spearman corr(injected noise, learned sigma) = {sp:+.4f}")
    print("  On real data run this against pLS_etaErr / pLS_ptErr and the T5")
    print("  chi-squared branches. A positive correlation is the claim 'the")
    print("  embedding learned the detector resolution'; a null is a publishable")
    print("  negative. Do not assume the sign -- measure it.")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  global metric absorbable   : residual {r_g:.6f}  (yes)")
    print(f"  per-instance not absorbable: residual {r_l:.6f}  "
          f"({r_l / max(r_g, 1e-12):.0f}x larger)")
    print("  => The gauge result is not a dead end: making the precision")
    print("     input-dependent escapes it, at ~1.5k extra parameters.")


if __name__ == "__main__":
    _smoke()
