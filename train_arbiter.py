"""
Train the arbitration head on top of a trained embedding.

Completes the pipeline:

    build_pairs.py   -> pairs.npz  (features + labels + per-candidate quality)
    metric_study.py  -> results/<metric>.pt  (trained encoders)
    train_arbiter.py -> results/arbiter.pt   (which duplicate to KEEP)

The embedding decides WHETHER two candidates are the same particle; the arbiter
decides WHICH ONE SURVIVES. Production CMSSW answers the second question with a
hard-coded collection priority, so the baseline this must beat is not another
network -- it is a fixed rule.

Two baselines are evaluated alongside the model, and reporting both is the point:

  random        -- 0.5 by construction, sanity check only
  quality-blind -- the accuracy a fixed priority rule achieves. Because the
                   arbiter only ever sees pairs of the SAME object type here
                   (T5 vs T5), a collection-priority rule has no signal at all
                   and degenerates to a coin flip. That is precisely the gap:
                   the production rule cannot arbitrate same-collection
                   duplicates, and same-collection duplicates are most of them.

Usage
-----
    python train_arbiter.py --pairs pairs.npz --encoder results/euclidean.pt
    python train_arbiter.py --synthetic          # smoke test, no data needed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from arbitration import PairwiseArbiter, antisymmetry_error, arbitration_accuracy
from metric_study import T5_IN_DIM, make_encoder


def load_duplicates(path: str, min_quality_gap: float):
    """Keep only genuine duplicate pairs with a decisive quality difference.

    Pairs whose two candidates are of equal quality carry no arbitration signal
    -- either choice is correct -- so training on them injects pure label noise.
    They are excluded from training and reported separately.
    """
    d = np.load(path)
    for k in ("q_left_t5", "q_right_t5"):
        if k not in d.files:
            raise SystemExit(
                f"{path} has no '{k}'. Regenerate pairs with build_pairs.py, "
                "which carries t5_pMatched through; the notebook's own pair "
                "cells discard it."
            )

    dup = d["y_t5"] == 0
    ql, qr = d["q_left_t5"][dup], d["q_right_t5"][dup]
    XL, XR = d["X_left"][dup], d["X_right"][dup]

    finite = np.isfinite(ql) & np.isfinite(qr)
    gap = np.abs(ql - qr)
    decisive = finite & (gap > min_quality_gap)

    n_tie = int((finite & ~decisive).sum())
    print(f"  duplicate pairs                 : {int(dup.sum()):,}")
    print(f"  with finite quality on both     : {int(finite.sum()):,}")
    print(f"  decisive (|dq| > {min_quality_gap})          : {int(decisive.sum()):,}")
    print(f"  ties, excluded from training    : {n_tie:,} "
          f"({n_tie / max(int(finite.sum()), 1):.1%} of usable pairs)")
    if decisive.sum() < 1000:
        print("  ** WARNING: very few decisive pairs. If t5_pMatched is near-"
              "constant\n     across duplicates, it is the wrong arbitration "
              "target and a different\n     quality definition is needed.")

    keep_left = (ql[decisive] > qr[decisive]).astype(np.float32)
    return (XL[decisive].astype(np.float32), XR[decisive].astype(np.float32),
            keep_left, gap[decisive].astype(np.float32))


def load_encoder(path: str, emb_dim: int, hidden: int, device):
    """Load the frozen T5 encoder trained by metric_study.py."""
    enc = make_encoder(T5_IN_DIM, emb_dim, hidden).to(device)
    sd = torch.load(path, map_location=device)
    enc.load_state_dict(sd["enc_t5"] if "enc_t5" in sd else sd)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def synthetic(n=20000, emb_dim=16, seed=0):
    rng = np.random.default_rng(seed)
    XL = rng.standard_normal((n, T5_IN_DIM)).astype(np.float32)
    XR = rng.standard_normal((n, T5_IN_DIM)).astype(np.float32)
    # quality is a hidden linear function of a few raw features
    w = rng.standard_normal(T5_IN_DIM)
    ql, qr = XL @ w, XR @ w
    keep = (ql > qr).astype(np.float32)
    return XL, XR, keep, np.abs(ql - qr).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--encoder", default=None, help="results/<metric>.pt from metric_study.py")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--min-quality-gap", type=float, default=1e-3)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ARBITRATION: which duplicate survives")
    print("=" * 70)

    if args.synthetic:
        XL, XR, keep, gap = synthetic()
        enc = make_encoder(T5_IN_DIM, args.emb_dim, args.hidden).to(device).eval()
        for p in enc.parameters():
            p.requires_grad = False
    else:
        if not (args.pairs and args.encoder):
            raise SystemExit("need --pairs and --encoder, or --synthetic")
        XL, XR, keep, gap = load_duplicates(args.pairs, args.min_quality_gap)
        enc = load_encoder(args.encoder, args.emb_dim, args.hidden, device)

    n = len(keep)
    idx = np.random.default_rng(0).permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]

    t = lambda a: torch.tensor(a, device=device)
    XL_t, XR_t, keep_t = t(XL), t(XR), t(keep)
    # weight by how decisive the pair is: a near-tie should not dominate the loss
    w_t = t(gap / (gap.mean() + 1e-9))

    with torch.no_grad():
        EL, ER = enc(XL_t), enc(XR_t)

    # ------------------------------------------------------------------
    # THE INFORMATION-DESTRUCTION ABLATION
    #
    # The contrastive loss explicitly minimises ||e_i - e_j|| over duplicate
    # pairs. In the limit where it succeeds perfectly, e_i == e_j, and the
    # antisymmetric score g(e_i,e_j) - g(e_j,e_i) is identically ZERO -- the
    # arbiter has no signal at all. So the better the embedding is at its own
    # job, the LESS arbitration information survives in it.
    #
    # This is a real tension, not a hypothetical, and it decides the
    # architecture: the arbiter must read raw features, not only the embedding.
    # Running all three inputs quantifies the cost.
    # ------------------------------------------------------------------
    def train_one(mode: str):
        if mode == "embedding":
            A, B, ed, xd = EL, ER, args.emb_dim, 0
            xa = xb = None
        elif mode == "raw":
            A, B, ed, xd = XL_t, XR_t, T5_IN_DIM, 0
            xa = xb = None
        else:  # both
            A, B, ed, xd = EL, ER, args.emb_dim, T5_IN_DIM
            xa, xb = XL_t, XR_t

        torch.manual_seed(0)
        m = PairwiseArbiter(ed, extra_dim=xd).to(device)
        o = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-5)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, args.epochs)

        for ep in range(1, args.epochs + 1):
            m.train()
            perm = torch.randperm(len(tr), device=device)
            tot, nb = 0.0, 0
            for s in range(0, len(tr), args.batch_size):
                b = t(tr)[perm[s:s + args.batch_size]]
                o.zero_grad()
                sc_b = m(A[b], B[b], None if xa is None else xa[b],
                         None if xb is None else xb[b])
                loss = PairwiseArbiter.loss(sc_b, keep_t[b], w_t[b])
                loss.backward()
                o.step()
                tot += loss.item()
                nb += 1
            sc.step()
            if ep % 10 == 0 or ep == 1:
                print(f"    [{mode:9s}] ep {ep:3d}  loss={tot / max(nb, 1):.4f}")

        m.eval()
        ti = t(te)
        with torch.no_grad():
            s_te = m(A[ti], B[ti], None if xa is None else xa[ti],
                     None if xb is None else xb[ti])
        sv = s_te.squeeze(-1).cpu().numpy()
        med = np.median(gap[te])
        hard = gap[te] > med
        return m, {
            "accuracy": arbitration_accuracy(s_te, keep_t[ti]),
            "accuracy_decisive_half": float(((sv > 0)[hard] == keep[te][hard]).mean()),
            "ranking_auc": float(roc_auc_score(keep[te], sv)),
            "antisymmetry_error": antisymmetry_error(
                m, A[ti], B[ti], None if xa is None else xa[ti],
                None if xb is None else xb[ti]),
        }

    print(f"\n  train {len(tr):,}  test {len(te):,}  device {device}")

    # how much do duplicate pairs actually collapse together in the embedding?
    with torch.no_grad():
        dd = (EL[t(te)] - ER[t(te)]).norm(dim=1)
        scale = torch.cat([EL[t(te)], ER[t(te)]]).std(0).norm()
    print(f"  mean ||e_i - e_j|| over duplicates : {float(dd.mean()):.4f}")
    print(f"  embedding scale (||std||)          : {float(scale):.4f}")
    print(f"  ratio                              : {float(dd.mean() / scale):.4f}"
          "   (-> 0 means the embedding has erased the difference)")

    results, models = {}, {}
    for mode in ("embedding", "raw", "both"):
        print(f"\n  --- input: {mode} ---")
        models[mode], results[mode] = train_one(mode)

    print("\n" + "-" * 70)
    print(f"  {'input':<12}{'accuracy':>10}{'decisive':>10}{'AUC':>9}{'antisym':>10}")
    for mode, r in results.items():
        print(f"  {mode:<12}{r['accuracy']:>10.4f}{r['accuracy_decisive_half']:>10.4f}"
              f"{r['ranking_auc']:>9.4f}{r['antisymmetry_error']:>10.1e}")
    print(f"  {'random':<12}{0.5:>10.4f}{0.5:>10.4f}{0.5:>9.4f}")
    print(f"  {'priority rule':<12}{0.5:>10.4f}{0.5:>10.4f}{0.5:>9.4f}"
          "   (no signal for same-collection pairs)")
    print("-" * 70)

    delta = results["both"]["accuracy"] - results["embedding"]["accuracy"]
    print(f"\n  raw features add {delta:+.4f} accuracy over the embedding alone.")
    if delta > 0.02:
        print("  -> The embedding HAS discarded arbitration-relevant information,")
        print("     exactly as the contrastive objective is designed to. The")
        print("     arbiter must read raw features; an embedding-only head")
        print("     under-performs by construction.")

    best = max(results, key=lambda m: results[m]["accuracy"])
    res = {"n_train": len(tr), "n_test": len(te), "by_input": results,
           "best_input": best, "encoder": args.encoder, "pairs": args.pairs,
           "dup_separation_ratio": float(dd.mean() / scale)}
    torch.save({"arbiter": models[best].state_dict(), "emb_dim": args.emb_dim,
                "input_mode": best}, out / "arbiter.pt")
    (out / "arbiter_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/arbiter.pt (best input: {best}) and "
          f"{out}/arbiter_results.json")

    if results[best]["accuracy"] < 0.55:
        print("\n  ** Even the best input is near chance. Most likely cause:")
        print("     t5_pMatched is near-constant across duplicate pairs, i.e. it")
        print("     is the wrong arbitration target. Check its distribution")
        print("     before concluding the method fails.")


if __name__ == "__main__":
    main()
