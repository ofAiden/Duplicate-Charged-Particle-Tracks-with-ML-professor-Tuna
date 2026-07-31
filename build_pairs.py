"""
Pair generation that carries a per-candidate QUALITY target through.

WHY THIS EXISTS
---------------
`arbitration.py` needs to know, for each duplicate pair, which of the two
candidates is the better reconstruction. `t5_pMatched` (the fraction of the
candidate's hits belonging to its matched sim track) is exactly that quantity
and is already in the notebook's `branches_list` -- but the pair-generation
cells discard it, keeping only the features, the sim index and the displacement
flag. Without it the arbitration head has no label.

SCOPE
-----
This module deliberately does NOT re-implement the ROOT loading or the feature
building (notebook cells 1-4). Those are already working and expensive to
re-verify. It replaces only the pair-generation step (cells 5-6), taking the
per-event arrays the notebook has already built and adding two things:

  * `q_left` / `q_right` -- the per-candidate quality, carried through
  * an explicit measurement of the BLOCKING RECALL CEILING (see below)

BLOCKING RECALL CEILING
-----------------------
The dR^2 < 0.02 window is a blocking step in the entity-resolution sense. Any
true duplicate pair whose members fall outside the window can never be
recovered by any downstream model, so it caps achievable duplicate recall. The
notebook never measures this, which means the reported AUC silently excludes an
unknown fraction of the real problem. `pair_stats()` measures it.

USAGE (in the notebook, replacing cells 5 and 6)
------------------------------------------------
    import build_pairs

    t5 = build_pairs.t5_pairs(
        features_per_event, sim_indices_per_event, displaced_per_event,
        quality_per_event=pmatched_per_event,      # <- see note below
        eta_max=eta_max,
    )
    cross = build_pairs.cross_pairs(
        pLS_features_per_event, pLS_sim_indices_per_event,
        features_per_event, sim_indices_per_event, displaced_per_event,
        quality_t5_per_event=pmatched_per_event, eta_max=eta_max,
    )
    build_pairs.save("pairs.npz", t5, cross)

`pmatched_per_event` must be collected in the T5 feature-building loop (cell 3)
alongside `disp_evt`, with one line:

    qual_evt.append(branches['t5_pMatched'][ev][i])

Self-test (no ROOT file needed):  .venv/bin/python build_pairs.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

DELTA_R2_CUT = 0.02
INVALID_SIM_IDX = -1
DISP_VXY_CUT = 0.1
PLS_ETA_SCALE = 4.0


@dataclass
class PairSet:
    """Pairs plus the per-candidate quality needed for arbitration."""

    X_left: np.ndarray
    X_right: np.ndarray
    y: np.ndarray            # 0 = duplicate, 1 = not
    w: np.ndarray            # sample weight (displaced upweighting)
    q_left: np.ndarray       # per-candidate quality, left member
    q_right: np.ndarray      # per-candidate quality, right member
    sim_idx: np.ndarray      # shared sim index for duplicates, -1 otherwise
    event_id: np.ndarray     # which event each pair came from
    cand_i: np.ndarray       # index of the left candidate within its event
    cand_j: np.ndarray       # index of the right candidate within its event
    # per-CANDIDATE table (not per pair): needed for track-level risk and for
    # survivor/arbitration bookkeeping, which the pair table alone cannot support
    cand_event: np.ndarray
    cand_index: np.ndarray
    cand_sim: np.ndarray
    cand_quality: np.ndarray
    stats: dict

    def __len__(self) -> int:
        return len(self.y)


def _phi_from_feats(F: np.ndarray, cos_col: int, sin_col: int) -> np.ndarray:
    return np.arctan2(F[:, sin_col], F[:, cos_col])


def _delta_r2(eta_a, phi_a, eta_b, phi_b):
    dphi = (phi_a - phi_b + np.pi) % (2 * np.pi) - np.pi
    return (eta_a - eta_b) ** 2 + dphi ** 2


# ----------------------------------------------------------------------------
def _t5_pairs_one_event(evt, F, S, D, Q, max_sim, max_dis, rng):
    n = F.shape[0]
    if n < 2:
        return None

    eta = F[:, 0] * _t5_pairs_one_event.eta_max
    phi = _phi_from_feats(F, 1, 2)

    i, j = np.triu_indices(n, k=1)
    dr2 = _delta_r2(eta[i], phi[i], eta[j], phi[j])
    in_window = dr2 < DELTA_R2_CUT

    same_sim = S[i] == S[j]
    valid_sim = S[i] != INVALID_SIM_IDX

    # true duplicates ANYWHERE, used to measure the blocking recall ceiling
    is_dup_any = same_sim & valid_sim
    n_dup_total = int(is_dup_any.sum())
    n_dup_in_window = int((is_dup_any & in_window).sum())

    sim_mask = in_window & is_dup_any
    dis_mask = in_window & ~same_sim

    idx = np.stack((i, j), axis=-1)
    sim_pairs, dis_pairs = idx[sim_mask], idx[dis_mask]
    sim_sidx = S[i][sim_mask]

    if len(sim_pairs) > max_sim:
        sel = rng.choice(len(sim_pairs), max_sim, replace=False)
        sim_pairs, sim_sidx = sim_pairs[sel], sim_sidx[sel]
    if len(dis_pairs) > max_dis:
        dis_pairs = dis_pairs[rng.choice(len(dis_pairs), max_dis, replace=False)]

    a = np.concatenate([sim_pairs[:, 0], dis_pairs[:, 0]])
    b = np.concatenate([sim_pairs[:, 1], dis_pairs[:, 1]])
    y = np.concatenate([np.zeros(len(sim_pairs), np.float32),
                        np.ones(len(dis_pairs), np.float32)])
    sidx = np.concatenate([sim_sidx, np.full(len(dis_pairs), INVALID_SIM_IDX, np.int64)])
    disp = (D[a] > DISP_VXY_CUT) | (D[b] > DISP_VXY_CUT)

    return {
        "X_left": F[a], "X_right": F[b], "y": y,
        "q_left": Q[a], "q_right": Q[b],
        "sim_idx": sidx,
        "event_id": np.full(len(y), evt, np.int64),
        "cand_i": a.astype(np.int64), "cand_j": b.astype(np.int64),
        "cand_event": np.full(n, evt, np.int64),
        "cand_index": np.arange(n, dtype=np.int64),
        "cand_sim": S.astype(np.int64), "cand_quality": Q.astype(np.float32),
        "w": np.where(disp, 5.0, 1.0).astype(np.float32),
        "n_dup_total": n_dup_total, "n_dup_in_window": n_dup_in_window,
        "n_cand": n,
    }


def t5_pairs(features_per_event, sim_indices_per_event, displaced_per_event,
             quality_per_event, *, eta_max, max_sim=1000, max_dis=1000,
             seed=42, verbose=True) -> PairSet:
    """T5-T5 pairs with quality carried through."""
    _t5_pairs_one_event.eta_max = eta_max
    t0 = time.time()
    out, n_dup_tot, n_dup_win, n_cand = [], 0, 0, 0

    for ev in range(len(features_per_event)):
        r = _t5_pairs_one_event(
            ev, features_per_event[ev], sim_indices_per_event[ev],
            displaced_per_event[ev], quality_per_event[ev],
            max_sim, max_dis, np.random.default_rng(seed + ev),
        )
        if r is None:
            continue
        n_dup_tot += r.pop("n_dup_total")
        n_dup_win += r.pop("n_dup_in_window")
        n_cand += r.pop("n_cand")
        out.append(r)

    ps = _assemble(out, n_dup_tot, n_dup_win, n_cand, time.time() - t0, "T5-T5")
    if verbose:
        _report(ps)
    return ps


# ----------------------------------------------------------------------------
def cross_pairs(pls_features_per_event, pls_sim_per_event,
                t5_features_per_event, t5_sim_per_event, t5_displaced_per_event,
                quality_t5_per_event, *, eta_max, quality_pls_per_event=None,
                max_sim=1000, max_dis=1000, seed=42, verbose=True) -> PairSet:
    """pLS-T5 pairs.

    NOTE ON THE CROSS-COLLECTION QUALITY TARGET: `t5_pMatched` has no exact pLS
    counterpart in the branches the notebook currently loads. Pass
    `quality_pls_per_event` explicitly once the physics side decides what the
    comparable quantity is (`pLS_score` is in the ROOT file but is not currently
    read, and its sign convention must be checked before use). Until then this
    fills pLS quality with NaN so that any arbitration head trained on
    cross-collection pairs fails loudly rather than silently learning from a
    fabricated target.
    """
    t0 = time.time()
    out, n_dup_tot, n_dup_win, n_cand = [], 0, 0, 0
    n_ev = min(len(pls_features_per_event), len(t5_features_per_event))

    for ev in range(n_ev):
        Fp, Sp = pls_features_per_event[ev], pls_sim_per_event[ev]
        Ft, St = t5_features_per_event[ev], t5_sim_per_event[ev]
        Dt, Qt = t5_displaced_per_event[ev], quality_t5_per_event[ev]
        Qp = (np.full(len(Fp), np.nan, np.float32) if quality_pls_per_event is None
              else quality_pls_per_event[ev])
        if len(Fp) == 0 or len(Ft) == 0:
            continue
        rng = np.random.default_rng(seed + ev)

        eta_p = Fp[:, 0] * PLS_ETA_SCALE
        phi_p = _phi_from_feats(Fp, 2, 3)
        eta_t = Ft[:, 0] * eta_max
        phi_t = _phi_from_feats(Ft, 1, 2)

        ip, it = (x.ravel() for x in np.indices((len(Fp), len(Ft))))
        dr2 = _delta_r2(eta_p[ip], phi_p[ip], eta_t[it], phi_t[it])
        in_window = dr2 < DELTA_R2_CUT

        same_sim = Sp[ip] == St[it]
        valid = Sp[ip] != INVALID_SIM_IDX
        is_dup_any = same_sim & valid
        n_dup_tot += int(is_dup_any.sum())
        n_dup_win += int((is_dup_any & in_window).sum())
        n_cand += len(Fp) + len(Ft)

        sim_mask = in_window & is_dup_any
        dis_mask = in_window & ~same_sim
        sp = np.column_stack((ip[sim_mask], it[sim_mask]))
        dp = np.column_stack((ip[dis_mask], it[dis_mask]))
        s_sidx = Sp[ip][sim_mask]

        if len(sp) > max_sim:
            sel = rng.choice(len(sp), max_sim, replace=False)
            sp, s_sidx = sp[sel], s_sidx[sel]
        if len(dp) > max_dis:
            dp = dp[rng.choice(len(dp), max_dis, replace=False)]

        a = np.concatenate([sp[:, 0], dp[:, 0]])
        b = np.concatenate([sp[:, 1], dp[:, 1]])
        y = np.concatenate([np.zeros(len(sp), np.float32), np.ones(len(dp), np.float32)])
        sidx = np.concatenate([s_sidx, np.full(len(dp), INVALID_SIM_IDX, np.int64)])
        disp = Dt[b] > DISP_VXY_CUT

        out.append({
            "X_left": Fp[a], "X_right": Ft[b], "y": y,
            "q_left": Qp[a], "q_right": Qt[b], "sim_idx": sidx,
            "event_id": np.full(len(y), ev, np.int64),
            "cand_i": a.astype(np.int64), "cand_j": b.astype(np.int64),
            "cand_event": np.full(len(Ft), ev, np.int64),
            "cand_index": np.arange(len(Ft), dtype=np.int64),
            "cand_sim": St.astype(np.int64), "cand_quality": Qt.astype(np.float32),
            "w": np.where(disp, 5.0, 1.0).astype(np.float32),
        })

    ps = _assemble(out, n_dup_tot, n_dup_win, n_cand, time.time() - t0, "pLS-T5")
    if verbose:
        _report(ps)
    return ps


# ----------------------------------------------------------------------------
def _assemble(chunks, n_dup_tot, n_dup_win, n_cand, elapsed, name) -> PairSet:
    if not chunks:
        raise ValueError(f"{name}: no pairs generated -- check filters and inputs")
    cat = lambda k: np.concatenate([c[k] for c in chunks])
    X_left, X_right = cat("X_left"), cat("X_right")

    finite = np.isfinite(X_left).all(1) & np.isfinite(X_right).all(1)
    n_bad = int((~finite).sum())

    ps = PairSet(
        X_left=X_left[finite].astype(np.float32),
        X_right=X_right[finite].astype(np.float32),
        y=cat("y")[finite].astype(np.float32),
        w=cat("w")[finite].astype(np.float32),
        q_left=cat("q_left")[finite].astype(np.float32),
        q_right=cat("q_right")[finite].astype(np.float32),
        sim_idx=cat("sim_idx")[finite].astype(np.int64),
        event_id=cat("event_id")[finite].astype(np.int64),
        cand_i=cat("cand_i")[finite].astype(np.int64),
        cand_j=cat("cand_j")[finite].astype(np.int64),
        cand_event=cat("cand_event").astype(np.int64),
        cand_index=cat("cand_index").astype(np.int64),
        cand_sim=cat("cand_sim").astype(np.int64),
        cand_quality=cat("cand_quality").astype(np.float32),
        stats={
            "name": name,
            "seconds": elapsed,
            "n_candidates": n_cand,
            "n_pairs_kept": int(finite.sum()),
            "n_events": int(len(np.unique(cat("event_id")[finite]))),
            "n_dropped_nonfinite": n_bad,
            "n_true_duplicate_pairs_total": n_dup_tot,
            "n_true_duplicate_pairs_in_window": n_dup_win,
            "blocking_recall_ceiling": (n_dup_win / n_dup_tot) if n_dup_tot else float("nan"),
        },
    )
    return ps


def _report(ps: PairSet) -> None:
    s = ps.stats
    print(f"\n[{s['name']}] {len(ps):,} pairs in {s['seconds']:.1f}s "
          f"({(ps.y == 0).mean():.1%} duplicates)")
    if s["n_dropped_nonfinite"]:
        print(f"  dropped {s['n_dropped_nonfinite']:,} pairs with NaN/Inf")
    print(f"  BLOCKING RECALL CEILING: {s['blocking_recall_ceiling']:.4f}")
    print(f"    {s['n_true_duplicate_pairs_in_window']:,} of "
          f"{s['n_true_duplicate_pairs_total']:,} true duplicate pairs fall inside "
          f"dR^2 < {DELTA_R2_CUT}.")
    print("    No model can recover the rest. Report this alongside any recall"
          "\n    number -- it is the ceiling every downstream metric sits under.")
    if np.isnan(ps.q_left).any() or np.isnan(ps.q_right).any():
        n = int(np.isnan(ps.q_left).sum() + np.isnan(ps.q_right).sum())
        print(f"  NOTE: {n:,} quality entries are NaN (no pLS counterpart to "
              "t5_pMatched yet).\n    Arbitration on these pairs must be skipped "
              "until a target is chosen.")


def save(path, t5: PairSet, cross: PairSet) -> None:
    """Write the npz that metric_study.py and arbitration.py consume."""
    np.savez_compressed(
        path,
        X_left=t5.X_left, X_right=t5.X_right, y_t5=t5.y, w_t5=t5.w,
        q_left_t5=t5.q_left, q_right_t5=t5.q_right, sim_idx_t5=t5.sim_idx,
        event_id_t5=t5.event_id, cand_i_t5=t5.cand_i, cand_j_t5=t5.cand_j,
        cand_event_t5=t5.cand_event, cand_index_t5=t5.cand_index,
        cand_sim_t5=t5.cand_sim, cand_quality_t5=t5.cand_quality,
        X_pls=cross.X_left, X_t5cross=cross.X_right, y_pls=cross.y, w_pls=cross.w,
        q_left_pls=cross.q_left, q_right_pls=cross.q_right, sim_idx_pls=cross.sim_idx,
        event_id_pls=cross.event_id, cand_i_pls=cross.cand_i,
        cand_j_pls=cross.cand_j,
        blocking_ceiling_t5=np.float32(t5.stats["blocking_recall_ceiling"]),
        blocking_ceiling_pls=np.float32(cross.stats["blocking_recall_ceiling"]),
    )
    print(f"\nwrote {path}")


# ----------------------------------------------------------------------------
def _self_test():
    """Synthetic events with a known number of in-window duplicates, so the
    blocking-ceiling arithmetic can be checked against ground truth."""
    rng = np.random.default_rng(0)
    eta_max = 2.5
    n_events = 12
    feats, sims, disps, quals = [], [], [], []

    for _ in range(n_events):
        n = rng.integers(40, 80)
        F = rng.standard_normal((n, 30)).astype(np.float32)
        eta = rng.uniform(-2, 2, n)
        phi = rng.uniform(-np.pi, np.pi, n)
        F[:, 0] = eta / eta_max
        F[:, 1], F[:, 2] = np.cos(phi), np.sin(phi)
        # force some genuine duplicates: clone a few candidates nearby, and
        # place a few far away so the ceiling is genuinely below 1
        S = np.arange(n, dtype=np.int64)
        for k in range(0, 12, 2):
            S[k + 1] = S[k]
            if k % 4 == 0:                       # near clone -> inside window
                F[k + 1, 0] = F[k, 0] + 0.01
                F[k + 1, 1], F[k + 1, 2] = F[k, 1], F[k, 2]
            else:                                 # far clone -> outside window
                F[k + 1, 0] = F[k, 0] + 0.9
        feats.append(F)
        sims.append(S)
        disps.append(rng.random(n).astype(np.float32) * 0.3)
        quals.append(rng.random(n).astype(np.float32))

    print("=" * 70)
    print("SELF-TEST (synthetic)")
    print("=" * 70)
    t5 = t5_pairs(feats, sims, disps, quals, eta_max=eta_max, max_sim=50, max_dis=50)

    assert t5.q_left.shape == t5.y.shape == t5.q_right.shape
    assert np.isfinite(t5.q_left).all(), "T5 quality must be finite"
    dup = t5.y == 0
    assert (t5.sim_idx[dup] >= 0).all(), "duplicates must carry a sim index"
    assert (t5.sim_idx[~dup] == INVALID_SIM_IDX).all()
    c = t5.stats["blocking_recall_ceiling"]
    assert 0.0 < c < 1.0, f"synthetic ceiling should be strictly between 0 and 1, got {c}"
    assert t5.cand_i.shape == t5.y.shape == t5.cand_j.shape
    assert (t5.cand_i != t5.cand_j).all(), "a pair must join two distinct candidates"
    print(f"\n  quality carried through: OK ({len(t5):,} pairs)")
    print("  candidate indices present: OK (needed for transitivity + "
          "candidate-level risk)")
    # the candidate table must cover every candidate referenced by a pair
    for ev in np.unique(t5.event_id):
        pm, cm = t5.event_id == ev, t5.cand_event == ev
        referenced = set(t5.cand_i[pm]) | set(t5.cand_j[pm])
        assert referenced <= set(t5.cand_index[cm]), (
            f"event {ev}: pairs reference candidates absent from the table")
    print(f"  candidate table complete: OK ({len(t5.cand_index):,} candidates)")
    print(f"  sim_idx consistency:     OK")
    print(f"  ceiling in (0,1):        OK ({c:.3f} -- half the planted clones "
          "were placed outside the window by construction)")

    npls = [rng.standard_normal((30, 10)).astype(np.float32) for _ in range(n_events)]
    for P in npls:
        P[:, 0] = rng.uniform(-2, 2, len(P)) / PLS_ETA_SCALE
        ph = rng.uniform(-np.pi, np.pi, len(P))
        P[:, 2], P[:, 3] = np.cos(ph), np.sin(ph)
    psims = [rng.integers(0, 60, 30).astype(np.int64) for _ in range(n_events)]
    cross = cross_pairs(npls, psims, feats, sims, disps, quals, eta_max=eta_max,
                        max_sim=50, max_dis=50)
    assert np.isnan(cross.q_left).all(), "pLS quality should be NaN until chosen"
    print(f"\n  cross-collection pLS quality correctly NaN: OK")
    print("\nall assertions passed")


if __name__ == "__main__":
    _self_test()
