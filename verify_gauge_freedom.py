"""
Verify (or refute) the claim that a globally learned Mahalanobis metric adds
ZERO expressive power over Euclidean, when the encoder ends in a trainable
linear layer.

THE ARGUMENT
------------
Write the Mahalanobis matrix as V = L^T L (always possible for PSD V). The
encoder in metric_study.py ends in `nn.Linear(hidden//2, emb_dim)` with weight W
and bias c, so for a pair (a, b):

    f(a) - f(b) = (W h_a + c) - (W h_b + c) = W (h_a - h_b)      [bias cancels]

    d_V(f(a), f(b))^2 = (f(a)-f(b))^T V (f(a)-f(b))
                      = || L W (h_a - h_b) ||^2

Since L W is just some emb_dim x (hidden//2) matrix, the model class
{encoder ending in W, metric V} equals the model class
{encoder ending in LW, metric I}. So a global V is a GAUGE FREEDOM, not a model
extension -- it is unidentifiable in every direction except overall scale.

WHY THIS MATTERS
----------------
It predicts the already-measured result (V came out 94.9% pure rescaling) from
first principles, and it is a distinct mechanism from the numerical one
diagnosed in diagnose_metric_degeneracy.py (a fixed 1e-3 ridge swamping a
tiny-scale covariance). Both are real; they reinforce each other.

WHAT COULD BREAK IT -- tested below
-----------------------------------
  1. A non-linear final layer (e.g. a terminal ReLU) breaks the bias
     cancellation and the linear absorption.
  2. Weight decay breaks it: ||W||^2 != ||LW||^2, so the two parameterisations
     are penalised differently even though they compute the same distances.
  3. Cosine distance breaks it: cosine is not a function of (f(a)-f(b)) alone.

Run:  .venv/bin/python verify_gauge_freedom.py
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
DEV = torch.device("cpu")


def mahal_sq(diff, V):
    return ((diff @ V) * diff).sum(-1)


def main():
    n, hid, emb = 4096, 64, 12

    # ---------------------------------------------------------------- claim 1
    print("=" * 74)
    print("1. ALGEBRAIC IDENTITY: d_V(Wx, Wy) == d_I(LWx, LWy)")
    print("=" * 74)
    h_a, h_b = torch.randn(n, hid), torch.randn(n, hid)
    W = torch.randn(emb, hid) / np.sqrt(hid)
    c = torch.randn(emb)

    # an arbitrary PSD V, deliberately strongly anisotropic
    A = torch.randn(emb, emb)
    V = A.T @ A + 0.05 * torch.eye(emb)
    L = torch.linalg.cholesky(V).T                     # V = L^T L

    f_a, f_b = h_a @ W.T + c, h_b @ W.T + c
    d_mahal = mahal_sq(f_a - f_b, V)

    W2 = L @ W                                          # absorb L into W
    g_a, g_b = h_a @ W2.T + (L @ c), h_b @ W2.T + (L @ c)
    d_eucl = ((g_a - g_b) ** 2).sum(-1)

    err = (d_mahal - d_eucl).abs().max() / d_mahal.abs().max()
    print(f"  max relative difference : {float(err):.3e}")
    print(f"  V anisotropy (cond)     : {float(torch.linalg.cond(V)):.1f}")
    print(f"  VERDICT: {'IDENTICAL -- claim CONFIRMED' if err < 1e-5 else 'DIFFERENT -- claim REFUTED'}")

    # ---------------------------------------------------------------- claim 2
    print("\n" + "=" * 74)
    print("2. DOES IT SURVIVE A TERMINAL ReLU?  (v2's encoder had one)")
    print("=" * 74)
    r_a, r_b = torch.relu(f_a), torch.relu(f_b)
    d_m_relu = mahal_sq(r_a - r_b, V)
    d_e_relu = ((torch.relu(g_a) - torch.relu(g_b)) ** 2).sum(-1)
    err2 = (d_m_relu - d_e_relu).abs().max() / d_m_relu.abs().max()
    print(f"  max relative difference : {float(err2):.3e}")
    print(f"  VERDICT: {'still identical' if err2 < 1e-5 else 'BROKEN -- a terminal ReLU makes V genuinely expressive'}")
    print("  -> So the gauge argument applies to metric_study.py's encoder (no")
    print("     terminal ReLU) but NOT to an encoder ending in ReLU.")

    # ---------------------------------------------------------------- claim 3
    print("\n" + "=" * 74)
    print("3. DOES WEIGHT DECAY BREAK THE EQUIVALENCE?")
    print("=" * 74)
    nW, nW2 = float((W ** 2).sum()), float((W2 ** 2).sum())
    print(f"  ||W||_F^2  = {nW:.4f}")
    print(f"  ||LW||_F^2 = {nW2:.4f}   ratio {nW2 / nW:.2f}x")
    print("  VERDICT: the two parameterisations compute IDENTICAL distances but")
    print("  carry different weight-decay penalties, so with weight_decay > 0")
    print("  they are not optimisation-equivalent. The gauge freedom is exact")
    print("  for the FUNCTION CLASS, approximate for the TRAINED MODEL.")
    print("  (metric_study.py uses weight_decay=1e-4, so state this caveat.)")

    # ---------------------------------------------------------------- claim 4
    print("\n" + "=" * 74)
    print("4. DOES IT APPLY TO COSINE DISTANCE?")
    print("=" * 74)
    cos_f = torch.nn.functional.cosine_similarity(f_a, f_b, dim=1)
    cos_g = torch.nn.functional.cosine_similarity(g_a, g_b, dim=1)
    err4 = (cos_f - cos_g).abs().max()
    print(f"  max absolute difference : {float(err4):.3e}")
    print(f"  VERDICT: {'same' if err4 < 1e-5 else 'DIFFERENT -- cosine is not absorbable'}")
    print("  -> Cosine is not a function of (f(a)-f(b)) alone, so it is a")
    print("     genuinely different metric. Euclidean-vs-Mahalanobis is a gauge")
    print("     choice; Euclidean-vs-Cosine is not.")

    # ---------------------------------------------------------------- claim 4b
    print("\n" + "=" * 74)
    print("4b. DOES L2 NORMALISATION KILL THE GAUGE?")
    print("=" * 74)
    print("  Matters because published HEP metric learning L2-normalises onto the")
    print("  unit hypersphere (Chan, arXiv:2605.14131), which would make it immune")
    print("  to this critique. Bouhsine (arXiv:2602.19393) claims the ambiguity")
    print("  vanishes identically under a unit-sphere constraint -- test it.")
    na_f = torch.nn.functional.normalize(f_a, dim=1)
    nb_f = torch.nn.functional.normalize(f_b, dim=1)
    na_g = torch.nn.functional.normalize(g_a, dim=1)
    nb_g = torch.nn.functional.normalize(g_b, dim=1)
    d_m_norm = mahal_sq(na_f - nb_f, V)
    d_e_norm = ((na_g - nb_g) ** 2).sum(-1)
    err4b = (d_m_norm - d_e_norm).abs().max() / d_m_norm.abs().max()
    print(f"  max relative difference : {float(err4b):.3e}")
    print(f"  VERDICT: {'still absorbable' if err4b < 1e-5 else 'BROKEN -- L2 normalisation defeats the absorption'}")
    print("  -> Normalisation does not commute with the linear map (L x / |L x| is")
    print("     not L(x/|x|)), so an L2-normalised embedding is NOT covered by the")
    print("     gauge argument. Scope the claim to unnormalised embeddings, which")
    print("     is what CMSSW's t5embdnn/plsembdnn actually use (no normalisation).")

    # ---------------------------------------------------------------- claim 5
    print("\n" + "=" * 74)
    print("5. EMPIRICAL: can Euclidean training MATCH an anisotropic-V target?")
    print("=" * 74)
    print("  Fit both parameterisations to the same pair labels and compare.")
    print("  If the gauge argument holds, final losses should agree closely.")

    y = (torch.rand(n) < 0.5).float()
    lat = torch.randn(n, 8)
    M = torch.randn(8, hid)
    x_a = lat @ M + 0.1 * torch.randn(n, hid)
    partner = torch.where(y[:, None] == 0, lat, torch.randn(n, 8))
    x_b = partner @ M + 0.1 * torch.randn(n, hid)

    def fit(use_V, epochs=400, wd=0.0):
        torch.manual_seed(1)
        lin = nn.Linear(hid, emb)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
        Vc = V.clone()
        for _ in range(epochs):
            opt.zero_grad()
            fa, fb = lin(x_a), lin(x_b)
            d2 = mahal_sq(fa - fb, Vc) if use_V else ((fa - fb) ** 2).sum(-1)
            d = torch.sqrt(d2.clamp(min=1e-12))
            loss = ((1 - y) * d ** 2 + y * torch.relu(2.0 - d) ** 2).mean()
            loss.backward()
            opt.step()
        return float(loss.detach())

    for wd, tag in ((0.0, "weight_decay=0"), (1e-4, "weight_decay=1e-4")):
        lv, le = fit(True, wd=wd), fit(False, wd=wd)
        print(f"  {tag:<20} Mahalanobis {lv:.5f}   Euclidean {le:.5f}   "
              f"gap {abs(lv - le):.5f}")

    print("\n" + "=" * 74)
    print("BOTTOM LINE")
    print("=" * 74)
    print("  A global Mahalanobis metric on top of a linear-output encoder is a")
    print("  reparameterisation of Euclidean, not a richer model. To make a")
    print("  learned metric genuinely expressive it must be INPUT-DEPENDENT")
    print("  (per-pair precision), which no fixed final layer can absorb.")


if __name__ == "__main__":
    main()
