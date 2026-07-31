"""
Event-level Conformal Risk Control for the duplicate-removal threshold.

WHAT WAS WRONG BEFORE
---------------------
`metric_study.conformal_threshold` calibrated a split-conformal quantile over
~10^6 individual PAIRS. Those pairs are not exchangeable: they are drawn from
only ~500 events, and all ~10^7 candidate pairs inside one event share the same
underlying collision, the same pileup, and largely the same tracks. Treating
them as n = 10^6 independent draws inflates the effective sample size by roughly
the pairs-per-event factor (~4000x) and the resulting "guarantee" is void.

The exchangeable unit is the EVENT. That is what this module calibrates on, and
it also upgrades the controlled quantity from a per-pair indicator to a
physics-level risk: the fraction of genuinely distinct track pairs in an event
that get wrongly merged.

CONFORMAL RISK CONTROL  (Angelopoulos, Bates, Fisch, Lei, Schuster,
arXiv:2208.02814, Theorem 2.1)
------------------------------------------------------------------------------
Given per-event losses L_i(lambda) that are

  (a) non-increasing in lambda,
  (b) bounded above by B,
  (c) satisfy L_i(lambda_max) <= alpha,

the choice

    lambda_hat = inf { lambda : (n/(n+1)) * Rhat_n(lambda) + B/(n+1) <= alpha }

where Rhat_n(lambda) = (1/n) sum_i L_i(lambda), guarantees

    E[ L_{n+1}(lambda_hat) ] <= alpha

over the draw of the calibration set and a fresh event. Distribution-free;
requires only exchangeability of events.

We parameterise by lambda = -threshold, so that increasing lambda means a
smaller distance threshold, fewer merges, and lower loss -- satisfying (a). At
lambda_max = 0 nothing is merged and the loss is exactly 0, satisfying (c).

THE FINITE-SAMPLE PENALTY IS THE HEADLINE NUMBER
------------------------------------------------
The B/(n+1) term is paid before any data is examined. With 500 events, ~250 for
calibration, and the naive bound B = 1, it is 1/251 = 0.00398 -- so an
alpha = 0.005 target has 80% of its risk budget consumed up front, and any
alpha below ~0.004 is simply uncertifiable at event level with this dataset.
`budget_report()` makes that explicit, and it answers a question CMS does not
currently have an answer to: how many events are needed to certify a given
efficiency-loss budget.

Run the self-test:  .venv/bin/python conformal_risk.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# per-event loss
# ----------------------------------------------------------------------------
def event_false_merge_losses(d, y, event_id, thresholds):
    """L_i(t) = fraction of genuinely DISTINCT pairs in event i merged at
    threshold t.

    y == 1 means the two candidates are NOT the same particle, so merging them
    destroys a real track. This is the physics risk being controlled.

    Returns an (n_events, n_thresholds) array, non-decreasing along the
    threshold axis, plus the event ids in row order.

    NOTE ON SCOPE: this is a pair-level PROXY, calibrated at event level. It
    over-counts, because merging two candidates of the same sim track is correct
    behaviour yet still registers here as a merge of a "distinct" pair only when
    the labels say distinct -- and more importantly it charges for false merges
    that cost no track at all. Prefer `track_destruction_losses` below, which
    measures the actual physics quantity; keep this one as the cheap check that
    needs no candidate table.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    order = np.argsort(thresholds)
    if not np.array_equal(order, np.arange(len(thresholds))):
        raise ValueError("thresholds must be sorted ascending")

    # CRC needs L_i(lambda_max) = 0, i.e. zero loss at the least aggressive
    # threshold. With thresholds starting at 0 that holds iff every distance is
    # non-negative -- true for any metric, but worth asserting, because a
    # negative distance from a numerical bug would silently void the guarantee.
    d = np.asarray(d, dtype=np.float64)
    if np.any(d < 0):
        raise ValueError(
            f"{int((d < 0).sum())} negative distances: a metric cannot be "
            "negative, and it breaks the CRC boundary condition L(lambda_max)=0"
        )
    if not np.all(np.isfinite(d)):
        raise ValueError("non-finite distances")

    distinct = y == 1
    evs = np.unique(event_id[distinct])
    out = np.zeros((len(evs), len(thresholds)), dtype=np.float64)

    for r, ev in enumerate(evs):
        dd = d[distinct & (event_id == ev)]
        if len(dd) == 0:
            continue
        # fraction of this event's distinct pairs with distance below each
        # threshold -- searchsorted on the sorted distances is exact and fast
        ds = np.sort(dd)
        out[r] = np.searchsorted(ds, thresholds, side="left") / len(ds)
    return out, evs


def track_destruction_losses(d, event_id, cand_i, cand_j, thresholds,
                             cand_event, cand_index, cand_sim, cand_quality,
                             invalid_sim=-1):
    """THE physics risk: fraction of sim tracks in an event that lose every
    representative.

    At threshold t, all pairs with d < t are merged. Connected components over
    those merges each keep exactly ONE survivor -- the highest-quality member,
    which is what an arbiter approximates. A sim track is DESTROYED if it had at
    least one candidate before merging and none of its candidates survives.

    This is strictly stronger than `event_false_merge_losses`: a false merge only
    matters if it actually costs a track. Two candidates of DIFFERENT sim tracks
    merged together destroys exactly one track (the loser), whereas two
    candidates of the SAME sim track merged is correct behaviour and destroys
    nothing -- a distinction the pair-level proxy cannot make.

    Monotonicity: raising t only ever adds merges, so components only coarsen and
    the destroyed set only grows. Non-decreasing in t, hence non-increasing in
    lambda = -t, satisfying the CRC precondition. Asserted in the self-test.

    Returns (n_events, n_thresholds) and the event ids in row order.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if np.any(d < 0) or not np.all(np.isfinite(d)):
        raise ValueError("distances must be non-negative and finite")

    events = np.unique(cand_event)
    out = np.zeros((len(events), len(thresholds)), dtype=np.float64)

    for r, ev in enumerate(events):
        cm = cand_event == ev
        idx, sim, qual = cand_index[cm], cand_sim[cm], cand_quality[cm]
        pos = {int(c): k for k, c in enumerate(idx)}
        real = sim != invalid_sim
        tracks = np.unique(sim[real])
        if len(tracks) == 0:
            continue

        pm = event_id == ev
        pi, pj, pd = cand_i[pm], cand_j[pm], d[pm]
        order = np.argsort(pd)          # merge in increasing distance
        pi, pj, pd = pi[order], pj[order], pd[order]

        # union-find, incrementally extended as the threshold rises, so the
        # whole threshold sweep costs one pass rather than one pass per threshold
        parent = list(range(len(idx)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        cursor = 0
        for k, t in enumerate(thresholds):
            while cursor < len(pd) and pd[cursor] < t:
                a, b = pos.get(int(pi[cursor])), pos.get(int(pj[cursor]))
                if a is not None and b is not None:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
                cursor += 1

            # one survivor per component: the highest-quality member
            best: dict[int, int] = {}
            for m in range(len(idx)):
                root = find(m)
                cur = best.get(root)
                if cur is None or qual[m] > qual[cur]:
                    best[root] = m
            survivors = set(best.values())

            surviving_sims = {int(sim[m]) for m in survivors if real[m]}
            destroyed = sum(1 for tr in tracks if int(tr) not in surviving_sims)
            out[r, k] = destroyed / len(tracks)

    return out, events


# ----------------------------------------------------------------------------
# the CRC procedure
# ----------------------------------------------------------------------------
@dataclass
class CRCResult:
    alpha: float
    threshold: float          # the calibrated distance threshold
    n_calib_events: int
    B: float
    penalty: float            # B / (n+1)
    empirical_risk: float     # Rhat_n at the chosen threshold
    certifiable: bool
    reason: str = ""


def crc_threshold(losses, thresholds, alpha, B=1.0) -> CRCResult:
    """Calibrate the distance threshold by Conformal Risk Control.

    `losses` is (n_events, n_thresholds), non-decreasing along axis 1.
    Returns the LARGEST threshold (most aggressive duplicate removal) whose
    upper-confidence risk still respects alpha.
    """
    losses = np.asarray(losses, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    n = losses.shape[0]
    penalty = B / (n + 1)

    if n == 0:
        return CRCResult(alpha, 0.0, 0, B, float("inf"), float("nan"), False,
                         "no calibration events")
    if penalty > alpha:
        return CRCResult(
            alpha, 0.0, n, B, penalty, float("nan"), False,
            f"finite-sample penalty B/(n+1) = {penalty:.5f} already exceeds "
            f"alpha = {alpha}. Uncertifiable at this alpha with {n} events; "
            f"need n > B/alpha - 1 = {int(np.ceil(B / alpha - 1))} events, "
            f"or a tighter B.",
        )

    Rhat = losses.mean(axis=0)                      # non-decreasing in threshold
    ucb = (n / (n + 1)) * Rhat + penalty
    ok = ucb <= alpha

    if not ok.any():
        return CRCResult(alpha, 0.0, n, B, penalty, float(Rhat[0]), False,
                         "no threshold on the grid satisfies the bound; "
                         "extend the grid toward 0")

    # largest satisfying threshold; ucb is monotone so this is the last True
    k = int(np.flatnonzero(ok)[-1])
    return CRCResult(alpha, float(thresholds[k]), n, B, penalty,
                     float(Rhat[k]), True)


def estimate_B(losses) -> float:
    """A defensible upper bound on the loss for the CRC formula.

    The loss is a fraction, so B = 1 is always valid but maximally pessimistic.
    Because the procedure never considers a threshold beyond the top of the
    grid, the worst loss attainable ON THAT GRID is a valid bound for the
    restricted procedure -- and it is usually far below 1.

    Caveat to state in the paper: estimating B from the calibration data is
    mildly circular. The clean version fixes B a priori from the grid's top
    threshold on a held-out split. `crc_threshold` accepts B explicitly so
    either choice can be made and reported.
    """
    return float(np.max(losses)) if losses.size else 1.0


# ----------------------------------------------------------------------------
# group-conditional (Mondrian) calibration
# ----------------------------------------------------------------------------
def mondrian_crc(losses_by_group, thresholds, alpha, B=1.0):
    """One certified threshold per group (e.g. per eta bin).

    Production CMSSW already deploys eta-binned working points
    (`dnn::plsembdnn::kWP[bin_idx]`) with no statistical guarantee. Mondrian
    conformal calibration keeps that structure and attaches a finite-sample
    certificate to each bin (arXiv:2107.07511, sec. 4.1).

    The catch worth reporting: splitting n events across G groups pays the
    B/(n_g + 1) penalty G times over, on G-times-smaller samples. With 250
    calibration events and 5 eta bins, each bin has ~50 events and a penalty of
    1/51 = 0.0196 at B = 1 -- four times the alpha you were targeting. Group
    conditioning is statistically more honest and practically more expensive,
    and that trade is a result in itself.
    """
    return {g: crc_threshold(L, thresholds, alpha, B)
            for g, L in losses_by_group.items()}


# ----------------------------------------------------------------------------
def budget_report(n_events_total, alphas=(0.001, 0.002, 0.005, 0.01),
                  calib_frac=0.5, Bs=(1.0, 0.1, 0.05)):
    """How much of the risk budget the finite-sample penalty consumes, and how
    many events would be needed to certify a given alpha."""
    n = int(n_events_total * calib_frac)
    print(f"\n{'=' * 74}")
    print(f"RISK BUDGET  ({n_events_total} events, {calib_frac:.0%} for "
          f"calibration -> n = {n})")
    print("=" * 74)
    print(f"  {'alpha':>8}{'B':>8}{'penalty':>11}{'% of budget':>13}"
          f"{'events needed':>15}")
    for a in alphas:
        for B in Bs:
            pen = B / (n + 1)
            need = int(np.ceil(B / a - 1))
            flag = "" if pen <= a else "  UNCERTIFIABLE"
            print(f"  {a:>8.4f}{B:>8.2f}{pen:>11.5f}{pen / a:>12.0%}"
                  f"{need:>15,}{flag}")
    print("\n  'events needed' is the minimum n for the penalty alone to fit")
    print("  inside alpha; the real requirement is strictly larger because the")
    print("  empirical risk must also fit. Tightening B is far cheaper than")
    print("  collecting events: B enters both the penalty AND the event count")
    print("  linearly.")


# ----------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(0)
    print("=" * 74)
    print("SELF-TEST: does event-level CRC actually control the risk?")
    print("=" * 74)

    n_events, pairs_per_event = 500, 4000
    thresholds = np.linspace(0.0, 4.0, 401)

    def make_events(n, shift=1.0):
        """Distinct pairs sit at larger distances than duplicates, with a
        per-event offset so that pairs within an event are correlated -- the
        exact structure that breaks pair-level calibration."""
        d, y, ev = [], [], []
        for i in range(n):
            off = rng.normal(0, 0.25)          # per-event nuisance
            k = pairs_per_event
            yy = (rng.random(k) < 0.5).astype(int)
            dd = np.where(
                yy == 1,
                rng.gamma(9.0, 0.25, k) * shift + off,   # distinct: far
                rng.gamma(2.0, 0.15, k) + off,           # duplicate: near
            )
            dd = np.clip(dd, 0.0, None)   # distances are non-negative
            d.append(dd); y.append(yy); ev.append(np.full(k, i))
        return np.concatenate(d), np.concatenate(y), np.concatenate(ev)

    d, y, ev = make_events(n_events)
    L, evs = event_false_merge_losses(d, y, ev, thresholds)
    print(f"  loss matrix: {L.shape}  (events x thresholds)")
    assert np.all(np.diff(L, axis=1) >= -1e-12), "loss must be non-decreasing in t"
    assert np.allclose(L[:, 0], 0.0), "loss at t=0 must be 0 (CRC boundary condition)"
    print("  monotone in threshold: OK")
    print("  L(t=0) == 0 (boundary condition): OK")

    # ---- coverage over repeated calibration/test splits
    print("\n  Coverage check: 300 random 50/50 event splits, alpha = 0.01")
    alpha = 0.01
    risks_crc, risks_naive = [], []
    for _ in range(300):
        perm = rng.permutation(len(evs))
        c, t = perm[: len(perm) // 2], perm[len(perm) // 2:]
        B = estimate_B(L[c])
        r = crc_threshold(L[c], thresholds, alpha, B)
        if not r.certifiable:
            continue
        k = int(np.searchsorted(thresholds, r.threshold))
        risks_crc.append(L[t][:, min(k, len(thresholds) - 1)].mean())

        # the WRONG way: pool all pairs, take the alpha-quantile of distinct-pair
        # distances, ignoring event structure
        mask_c = np.isin(ev, evs[c]) & (y == 1)
        thr_naive = np.quantile(d[mask_c], alpha)
        kn = int(np.searchsorted(thresholds, thr_naive))
        risks_naive.append(L[t][:, min(kn, len(thresholds) - 1)].mean())

    rc, rn = np.array(risks_crc), np.array(risks_naive)
    print(f"    event-level CRC : mean risk {rc.mean():.5f}  "
          f"p95 {np.quantile(rc, 0.95):.5f}  target <= {alpha}  "
          f"{'OK' if rc.mean() <= alpha else '** VIOLATED **'}")
    print(f"    pooled-pair     : mean risk {rn.mean():.5f}  "
          f"p95 {np.quantile(rn, 0.95):.5f}  target <= {alpha}  "
          f"{'OK' if rn.mean() <= alpha else '** VIOLATED **'}")
    print(f"    certified thresholds in {len(rc)}/300 splits")

    # ---- the penalty wall
    print("\n  The penalty wall (B = 1, so penalty = 1/(n+1)):")
    for a in (0.02, 0.01, 0.005, 0.002):
        r = crc_threshold(L[: n_events // 2], thresholds, a, B=1.0)
        state = f"thr={r.threshold:.3f}" if r.certifiable else "UNCERTIFIABLE"
        print(f"    alpha={a:<7} penalty={r.penalty:.5f}  {state}")
    print("\n  Tightening B by CAPPING THE THRESHOLD GRID:")
    print("    B is the worst loss reachable on the grid, so a grid that runs to")
    print("    a threshold merging everything gives B ~ 1 and buys nothing. Cap")
    print("    the grid at a threshold you would never deploy past, and B falls.")
    Lc = L[: n_events // 2]
    for t_cap in (4.0, 1.0, 0.5, 0.3):
        m = thresholds <= t_cap
        Bc = estimate_B(Lc[:, m])
        r = crc_threshold(Lc[:, m], thresholds[m], 0.005, B=Bc)
        state = f"thr={r.threshold:.3f}" if r.certifiable else "UNCERTIFIABLE"
        print(f"    grid cap {t_cap:<5} B={Bc:.4f}  penalty={r.penalty:.5f}  "
              f"alpha=0.005  {state}")
    print("    The cap must be justified a priori, not chosen after seeing the")
    print("    risk -- otherwise B is fitted to the calibration data and the")
    print("    guarantee is no longer distribution-free.")

    # ---- Mondrian
    print("\n  Mondrian (group-conditional) calibration, 5 groups:")
    groups = {f"bin{g}": L[g::5] for g in range(5)}
    for g, r in mondrian_crc(groups, thresholds, 0.01, B=1.0).items():
        state = f"thr={r.threshold:.3f}" if r.certifiable else "UNCERTIFIABLE"
        print(f"    {g}  n={r.n_calib_events:<4} penalty={r.penalty:.5f}  {state}")

    # ---- the stronger candidate-level risk
    print("\n" + "=" * 74)
    print("TRACK-DESTRUCTION RISK (candidate-level, the physics quantity)")
    print("=" * 74)
    n_ev, n_cand = 300, 30
    E, I, J, DD = [], [], [], []
    CE, CI, CS, CQ = [], [], [], []
    for e in range(n_ev):
        sim = rng.integers(0, n_cand // 2, n_cand)      # some tracks share candidates
        qual = rng.random(n_cand)
        CE += [e] * n_cand; CI += list(range(n_cand))
        CS += list(sim); CQ += list(qual)
        for a in range(n_cand):
            for b in range(a + 1, n_cand):
                if rng.random() > 0.3:
                    continue
                same = sim[a] == sim[b]
                # same-track pairs sit closer, as a trained embedding would place them
                dist = rng.gamma(2.0, 0.15) if same else rng.gamma(9.0, 0.25)
                E.append(e); I.append(a); J.append(b); DD.append(max(dist, 0.0))
    E, I, J, DD = map(np.array, (E, I, J, DD))
    CE, CI, CS, CQ = (np.array(CE), np.array(CI), np.array(CS),
                      np.array(CQ, dtype=np.float32))

    thr2 = np.linspace(0.0, 3.0, 121)
    Ltd, evs_td = track_destruction_losses(DD, E, I, J, thr2, CE, CI, CS, CQ)
    print(f"  loss matrix {Ltd.shape}")
    assert np.all(np.diff(Ltd, axis=1) >= -1e-12), "must be non-decreasing in t"
    assert np.allclose(Ltd[:, 0], 0.0), "no merges at t=0 => nothing destroyed"
    print("  monotone in threshold: OK")
    print("  L(t=0) == 0 (CRC boundary condition): OK")
    print(f"  mean destruction at t=0.5 / 1.0 / 2.0 : "
          f"{Ltd[:, 20].mean():.4f} / {Ltd[:, 40].mean():.4f} / {Ltd[:, 80].mean():.4f}")

    Btd = estimate_B(Ltd[: n_ev // 2])
    for a in (0.01, 0.02):
        r = crc_threshold(Ltd[: n_ev // 2], thr2, a, B=Btd)
        state = f"thr={r.threshold:.3f}" if r.certifiable else f"UNCERTIFIABLE ({r.reason[:44]})"
        print(f"  alpha={a:<6} B={Btd:.3f}  penalty={r.penalty:.5f}  {state}")
    print("\n  Contrast with the pair-level proxy: merging two candidates of the")
    print("  SAME sim track is correct and destroys nothing, which the pair-level")
    print("  false-merge rate counts as an error. This risk does not.")

    budget_report(500)
    print("\nself-test complete")


if __name__ == "__main__":
    _self_test()
