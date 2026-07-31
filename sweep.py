"""
Two experiments the paper needs and does not yet have.

(A) EMBEDDING-DIMENSION SWEEP -- the central claim, actually measured.
    The draft asserts that effective rank saturates regardless of the nominal
    embedding dimension, and that this explains the reported null result for
    raising the dimension from 6 to 32. That was asserted from a single 12-dim
    run. This sweeps d in {4,6,8,12,16,32} and measures RankMe and AUC at each.

    The claim is FALSIFIABLE and this is the falsification test:
      * if RankMe saturates near a fixed value while d grows, the claim holds;
      * if RankMe tracks d, the claim is wrong and must be removed.

(B) DISJOINT-HALF REPLICATION -- robustness.
    A genuinely different sample (other pileup, other process) is not available,
    so the next best control is two DISJOINT halves of the events, trained
    independently. This cannot detect a bias shared by the whole sample, and is
    reported as what it is: a check on subset and seed stability, not external
    validity.

Usage:
    python sweep.py --pairs pairs.npz --mode dims  --out sweep_dims.json
    python sweep.py --pairs pairs.npz --mode halves --out sweep_halves.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import metric_study as ms


def subset(arrays, mask_t5, mask_pls):
    out = dict(arrays)
    for k in ("X_left", "X_right", "y_t5", "w_t5"):
        out[k] = arrays[k][mask_t5]
    for k in ("X_pls", "X_t5cross", "y_pls", "w_pls"):
        out[k] = arrays[k][mask_pls]
    return out


def run_one(arrays, cfg, metrics, device, tag):
    tr_t5, te_t5, tr_pl, te_pl = ms.make_loaders(arrays, cfg, device)
    res = {}
    for name in metrics:
        t0 = time.time()
        br = ms.MetricBranch(name, cfg, device)
        last = {}
        for ep in range(1, cfg.epochs + 1):
            stats, embs, embs_cross = ms.run_epoch(br, tr_t5, tr_pl, cfg, train=True)
            if name == "mahalanobis" and ep % cfg.cov_update_interval == 0:
                if embs is not None:
                    br.metric.update(embs[: cfg.max_cov_samples])
                if embs_cross is not None:
                    br.metric_cross.update(embs_cross[: cfg.max_cov_samples])
            br.sched.step(stats["loss"])
            last = stats
        d_t5, y_t5 = ms.collect_distances(br, te_t5, cross=False)
        d_pl, y_pl = ms.collect_distances(br, te_pl, cross=True)
        from sklearn.metrics import roc_auc_score

        # RankMe on a large held-out embedding sample, not one training batch
        br.eval()
        with torch.no_grad():
            n = min(40000, len(arrays["X_left"]))
            E = br.enc_t5(torch.tensor(arrays["X_left"][:n]).to(device))
            Ep = br.enc_pls(torch.tensor(arrays["X_pls"][:n]).to(device))
        res[name] = {
            "emb_dim": cfg.emb_dim,
            "auc_t5": float(roc_auc_score(y_t5, d_t5)),
            "auc_pls": float(roc_auc_score(y_pl, d_pl)),
            "rankme_t5": ms.rankme(E),
            "rankme_pls": ms.rankme(Ep),
            "var_top3_t5": ms.variance_explained(E, 3),
            "margin_active": last.get("margin_active", float("nan")),
            "margin": br.margin,
            "seconds": time.time() - t0,
        }
        r = res[name]
        print(f"  [{tag}] {name:<12} d={cfg.emb_dim:<3} "
              f"AUC {r['auc_t5']:.4f}/{r['auc_pls']:.4f}  "
              f"RankMe {r['rankme_t5']:5.2f}/{cfg.emb_dim} "
              f"(pLS {r['rankme_pls']:5.2f})  var@3 {r['var_top3_t5']*100:5.1f}%  "
              f"active {r['margin_active']:.2f}  [{r['seconds']:.0f}s]")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.npz")
    ap.add_argument("--mode", choices=("dims", "halves"), default="dims")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-pairs", type=int, default=400000)
    ap.add_argument("--out", default="sweep.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = {k: v for k, v in np.load(a.pairs).items()}
    ceil_t5 = float(d.pop("blocking_ceiling_t5", np.nan))
    d.pop("blocking_ceiling_pls", None)
    for k in list(d):
        if k.startswith(("cand_", "event_id", "sim_idx", "q_left", "q_right")):
            d.pop(k)

    # subsample for tractability; same subset for every configuration so the
    # comparison across dimensions is not confounded by different data
    rng = np.random.default_rng(0)
    n_t5 = min(a.max_pairs, len(d["y_t5"]))
    n_pl = min(a.max_pairs, len(d["y_pls"]))
    i_t5 = rng.choice(len(d["y_t5"]), n_t5, replace=False)
    i_pl = rng.choice(len(d["y_pls"]), n_pl, replace=False)
    d = subset(d, i_t5, i_pl)
    print(f"device {device}  pairs T5 {n_t5:,}  pLS {n_pl:,}  epochs {a.epochs}")

    out = {"mode": a.mode, "epochs": a.epochs, "n_pairs_t5": n_t5,
           "blocking_ceiling_t5": ceil_t5, "runs": {}}

    if a.mode == "dims":
        print("\n=== (A) EMBEDDING-DIMENSION SWEEP ===")
        print("Claim under test: RankMe saturates while nominal dimension grows.\n")
        for dim in (4, 6, 8, 12, 16, 32):
            cfg = ms.Config(emb_dim=dim, epochs=a.epochs,
                            metrics=("euclidean", "cosine"))
            out["runs"][str(dim)] = run_one(d, cfg, cfg.metrics, device, f"d={dim}")
    else:
        print("\n=== (B) DISJOINT-HALF REPLICATION ===")
        print("Not a second sample: same events, disjoint halves. Detects subset")
        print("and seed instability only, NOT a bias shared by the whole sample.\n")
        h_t5 = np.arange(len(d["y_t5"])) < len(d["y_t5"]) // 2
        h_pl = np.arange(len(d["y_pls"])) < len(d["y_pls"]) // 2
        for half, (m5, mp) in enumerate([(h_t5, h_pl), (~h_t5, ~h_pl)]):
            cfg = ms.Config(emb_dim=12, epochs=a.epochs, seed=half,
                            metrics=("euclidean", "cosine"))
            torch.manual_seed(half)
            out["runs"][f"half{half}"] = run_one(
                subset(d, m5, mp), cfg, cfg.metrics, device, f"half{half}")

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")

    if a.mode == "dims":
        print("\n=== VERDICT ON THE SATURATION CLAIM ===")
        for m in ("euclidean", "cosine"):
            dims = sorted(int(k) for k in out["runs"])
            rk = [out["runs"][str(x)][m]["rankme_t5"] for x in dims]
            print(f"  {m:<10} " + "  ".join(f"d={x}:{r:.2f}" for x, r in zip(dims, rk)))
            growth = rk[-1] / max(rk[0], 1e-9)
            dim_growth = dims[-1] / dims[0]
            print(f"    nominal dim x{dim_growth:.0f}  ->  RankMe x{growth:.2f}"
                  f"   ({'SATURATES' if growth < 0.35 * dim_growth else 'TRACKS DIMENSION'})")


if __name__ == "__main__":
    main()
