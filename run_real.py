"""
Build features + pairs from the real ROOT file.

Replicates updated_fixed-3.ipynb cells 1-4 (ROOT load, T5 features, pLS
features) and then calls build_pairs.py, which adds the three things the
notebook's own pair cells do not produce:

  * t5_pMatched carried through as a per-candidate quality target
  * the blocking recall ceiling of the dR^2 < 0.02 window
  * per-candidate tables (sim index, quality) for track-level risk

Feature definitions are copied verbatim from the notebook so the numbers stay
comparable; any change here would silently invalidate the comparison with the
existing results.

Usage:
    python run_real.py --events 500 --out pairs.npz
    python run_real.py --events 25 --out pairs_small.npz   # fast smoke run
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import uproot

import build_pairs

BRANCHES = [
    't5_innerRadius', 't5_bridgeRadius', 't5_outerRadius', 't5_pt', 't5_eta',
    't5_phi', 't5_isFake', 't5_t3_idx0', 't5_t3_idx1',
    't5_t3_fakeScore1', 't5_t3_promptScore1', 't5_t3_displacedScore1',
    't5_t3_fakeScore2', 't5_t3_promptScore2', 't5_t3_displacedScore2',
    't5_pMatched', 't5_sim_vxy', 't5_sim_vz', 't5_matched_simIdx',
    'pLS_eta', 'pLS_etaErr', 'pLS_phi', 'pLS_matched_simIdx',
    'pLS_circleCenterX', 'pLS_circleCenterY', 'pLS_circleRadius',
    'pLS_ptIn', 'pLS_ptErr', 'pLS_px', 'pLS_py', 'pLS_pz',
    'pLS_isQuad', 'pLS_isFake',
]
BRANCHES += [f't5_t3_{i}_{s}' for i in (0, 2, 4)
             for s in ('r', 'z', 'eta', 'phi', 'layer')]


def delta_phi(p1, p2):
    d = p1 - p2
    if d > np.pi:
        d -= 2 * np.pi
    elif d < -np.pi:
        d += 2 * np.pi
    return d


def load(path, n_events):
    print(f"opening {path}")
    t0 = time.time()
    with uproot.open(path) as f:
        tree = f["tree"]
        total = tree.num_entries
        n = min(n_events, total)
        print(f"  {total} events in file, reading {n}")
        br = {}
        for b in BRANCHES:
            try:
                br[b] = tree[b].array(library="np", entry_stop=n)
            except Exception as e:
                print(f"  !! missing branch {b}: {type(e).__name__}")
    print(f"  loaded in {time.time() - t0:.1f}s")
    return br, n


def build_t5(br, n_events, z_max, r_max, eta_max, pmatched_cut=0.0):
    """Notebook cell 3, verbatim, plus the quality array it discards."""
    feats, etas, disps, sims, quals = [], [], [], [], []
    kept = init = 0
    for ev in range(n_events):
        n_t5 = len(br['t5_t3_idx0'][ev])
        init += n_t5
        if n_t5 == 0:
            continue
        fe, et, sm, dp, ql = [], [], [], [], []
        for i in range(n_t5):
            if br['t5_pMatched'][ev][i] < pmatched_cut:
                continue
            i0 = br['t5_t3_idx0'][ev][i]
            i1 = br['t5_t3_idx1'][ev][i]

            eta1 = br['t5_t3_0_eta'][ev][i0]
            eta2 = abs(br['t5_t3_2_eta'][ev][i0])
            eta3 = abs(br['t5_t3_4_eta'][ev][i0])
            eta4 = abs(br['t5_t3_2_eta'][ev][i1])
            eta5 = abs(br['t5_t3_4_eta'][ev][i1])

            p1 = br['t5_t3_0_phi'][ev][i0]; p2 = br['t5_t3_2_phi'][ev][i0]
            p3 = br['t5_t3_4_phi'][ev][i0]; p4 = br['t5_t3_2_phi'][ev][i1]
            p5 = br['t5_t3_4_phi'][ev][i1]

            z1 = abs(br['t5_t3_0_z'][ev][i0]); z2 = abs(br['t5_t3_2_z'][ev][i0])
            z3 = abs(br['t5_t3_4_z'][ev][i0]); z4 = abs(br['t5_t3_2_z'][ev][i1])
            z5 = abs(br['t5_t3_4_z'][ev][i1])

            r1 = br['t5_t3_0_r'][ev][i0]; r2 = br['t5_t3_2_r'][ev][i0]
            r3 = br['t5_t3_4_r'][ev][i0]; r4 = br['t5_t3_2_r'][ev][i1]
            r5 = br['t5_t3_4_r'][ev][i1]

            inR = br['t5_innerRadius'][ev][i]
            brR = br['t5_bridgeRadius'][ev][i]
            ouR = br['t5_outerRadius'][ev][i]

            s1f = br['t5_t3_fakeScore1'][ev][i]
            s1p = br['t5_t3_promptScore1'][ev][i]
            s1d = br['t5_t3_displacedScore1'][ev][i]
            df = br['t5_t3_fakeScore2'][ev][i] - s1f
            dp_ = br['t5_t3_promptScore2'][ev][i] - s1p
            dd = br['t5_t3_displacedScore2'][ev][i] - s1d

            fe.append([
                eta1 / eta_max, np.cos(p1), np.sin(p1), z1 / z_max, r1 / r_max,
                eta2 - abs(eta1), delta_phi(p2, p1), (z2 - z1) / z_max, (r2 - r1) / r_max,
                eta3 - eta2, delta_phi(p3, p2), (z3 - z2) / z_max, (r3 - r2) / r_max,
                eta4 - eta3, delta_phi(p4, p3), (z4 - z3) / z_max, (r4 - r3) / r_max,
                eta5 - eta4, delta_phi(p5, p4), (z5 - z4) / z_max, (r5 - r4) / r_max,
                1.0 / inR, 1.0 / brR, 1.0 / ouR,
                s1f, s1p, s1d, df, dp_, dd,
            ])
            et.append(eta1)
            dp.append(br['t5_sim_vxy'][ev][i])
            ql.append(br['t5_pMatched'][ev][i])       # <-- the arbitration target
            sl = br['t5_matched_simIdx'][ev][i]
            sm.append(sl[0] if len(sl) else -1)

        if fe:
            feats.append(np.asarray(fe, np.float32))
            etas.append(np.asarray(et, np.float32))
            disps.append(np.asarray(dp, np.float32))
            sims.append(np.asarray(sm, np.int64))
            quals.append(np.asarray(ql, np.float32))
            kept += len(fe)
        if (ev + 1) % 50 == 0:
            print(f"    T5 event {ev+1}/{n_events}  kept {kept:,}")
    print(f"  T5: kept {kept:,}/{init:,} ({kept/max(init,1)*100:.1f}%)")
    return feats, etas, disps, sims, quals


def build_pls(br, n_events, keep_frac=0.40, seed=42):
    """Notebook cell 4, verbatim (seeded, so the subsample is reproducible)."""
    rng = np.random.default_rng(seed)
    feats, etas, sims = [], [], []
    kept = init = 0
    for ev in range(n_events):
        n_p = len(br['pLS_eta'][ev])
        init += n_p
        if n_p == 0:
            continue
        fe, et, sm = [], [], []
        for i in range(n_p):
            if br['pLS_isFake'][ev][i]:
                continue
            if rng.random() > keep_frac:
                continue
            eta = br['pLS_eta'][ev][i]
            fe.append([
                eta / 4.0,
                br['pLS_etaErr'][ev][i] / .00139,
                np.cos(br['pLS_phi'][ev][i]), np.sin(br['pLS_phi'][ev][i]),
                1.0 / br['pLS_ptIn'][ev][i],
                np.log10(br['pLS_ptErr'][ev][i]),
                br['pLS_isQuad'][ev][i],
                np.log10(np.abs(br['pLS_circleCenterX'][ev][i])),
                np.log10(np.abs(br['pLS_circleCenterY'][ev][i])),
                np.log10(br['pLS_circleRadius'][ev][i]),
            ])
            et.append(eta)
            sl = br['pLS_matched_simIdx'][ev][i]
            sm.append(sl[0] if len(sl) else -1)
        if fe:
            feats.append(np.asarray(fe, np.float32))
            etas.append(np.asarray(et, np.float32))
            sims.append(np.asarray(sm, np.int64))
            kept += len(fe)
        if (ev + 1) % 50 == 0:
            print(f"    pLS event {ev+1}/{n_events}  kept {kept:,}")
    print(f"  pLS: kept {kept:,}/{init:,} ({kept/max(init,1)*100:.1f}%)")
    return feats, etas, sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_file.root")
    ap.add_argument("--events", type=int, default=500)
    ap.add_argument("--out", default="pairs.npz")
    ap.add_argument("--max-sim", type=int, default=1000)
    ap.add_argument("--max-dis", type=int, default=1000)
    args = ap.parse_args()

    br, n = load(args.root, args.events)

    z_max = max(np.max(e) for e in br['t5_t3_4_z'] if len(e))
    r_max = max(np.max(e) for e in br['t5_t3_4_r'] if len(e))
    eta_max, = (2.5,)
    print(f"  z_max {z_max:.3f}  r_max {r_max:.3f}  eta_max {eta_max}")

    print("\nbuilding T5 features ...")
    t5_f, _, t5_d, t5_s, t5_q = build_t5(br, n, z_max, r_max, eta_max)
    print("\nbuilding pLS features ...")
    pls_f, _, pls_s = build_pls(br, n)

    # quality diagnostic -- decides whether the arbitration head has a label
    allq = np.concatenate(t5_q)
    print(f"\nt5_pMatched over {len(allq):,} candidates:")
    print(f"  min {allq.min():.4f}  median {np.median(allq):.4f}  "
          f"max {allq.max():.4f}  std {allq.std():.4f}")
    print(f"  unique values: {len(np.unique(allq)):,}")

    print("\ngenerating T5-T5 pairs ...")
    t5 = build_pairs.t5_pairs(t5_f, t5_s, t5_d, t5_q, eta_max=eta_max,
                              max_sim=args.max_sim, max_dis=args.max_dis)
    print("\ngenerating pLS-T5 pairs ...")
    cross = build_pairs.cross_pairs(pls_f, pls_s, t5_f, t5_s, t5_d, t5_q,
                                    eta_max=eta_max, max_sim=args.max_sim,
                                    max_dis=args.max_dis)
    build_pairs.save(args.out, t5, cross)


if __name__ == "__main__":
    main()
