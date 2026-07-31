"""
Global consistency of pairwise duplicate decisions.

THE PROBLEM
-----------
Duplicate removal is decided PAIRWISE, but "is the same particle as" is an
equivalence relation and must be transitive. A pairwise scorer has no mechanism
to enforce that, so it can assert a~b and b~c while denying a~c. Binette &
Steorts (arXiv:2008.04443) state it directly: pairwise methods "treat record
pairs as being independent of one another, without accounting for the
consequences of transitivity or other constraints on the linkage structure."

Nobody has measured this for LST. It costs one pass over the pair graph, it is
a property of the CURRENT production method (which thresholds a symmetric
distance pairwise, exactly as here), and it motivates any global clustering step.

WHY IT MATTERS PRACTICALLY
-------------------------
Hassanzadeh et al. (PVLDB 2(1), 2009) quantified how badly naive resolution
degrades under a permissive threshold: transitive closure chained 5,000 records
into 51 clusters at precision 0.104, while Markov clustering held 323 clusters at
precision 0.571. The LST embedding measured an RMS norm of 0.062 against a margin
of 1.0, i.e. all distances sit in a narrow band -- structurally the fragile
permissive-threshold regime.

Run:  .venv/bin/python transitivity.py
"""

from __future__ import annotations

import numpy as np


def intransitivity_rate(event_id, cand_i, cand_j, is_dup, max_events=None):
    """Fraction of connected triples (a,b,c) that violate transitivity.

    A triple is COUNTED only if all three pairs (a,b), (b,c), (a,c) were
    actually scored -- i.e. all three are present in the pair table. Triples
    with a missing edge are excluded, because the dR window never presented that
    pair and the model was never asked. Conflating "denied" with "never asked"
    would inflate the rate, and the missing-edge count is reported separately.

    A counted triple VIOLATES transitivity if exactly two of its three pairs are
    marked duplicate: a~b and b~c but not a~c.
    """
    events = np.unique(event_id)
    if max_events is not None:
        events = events[:max_events]

    n_closed = n_viol = n_open = 0
    n_dup_edges_total = 0

    for ev in events:
        m = event_id == ev
        ii, jj, dd = cand_i[m], cand_j[m], is_dup[m]
        # undirected edge -> decision lookup, plus the "was it scored" set
        scored = {}
        for a, b, d in zip(ii, jj, dd):
            key = (a, b) if a < b else (b, a)
            scored[key] = bool(d)
        n_dup_edges_total += sum(scored.values())

        # adjacency over DUPLICATE edges only; a violation needs two dup edges
        adj: dict[int, set[int]] = {}
        for (a, b), d in scored.items():
            if d:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

        # enumerate paths a-b-c through duplicate edges, then inspect (a,c)
        for b, nbrs in adj.items():
            nb = sorted(nbrs)
            for x in range(len(nb)):
                for y in range(x + 1, len(nb)):
                    a, c = nb[x], nb[y]
                    key = (a, c) if a < c else (c, a)
                    if key not in scored:
                        n_open += 1          # never presented to the model
                        continue
                    n_closed += 1
                    if not scored[key]:
                        n_viol += 1

    return {
        "n_events": len(events),
        "n_duplicate_edges": n_dup_edges_total,
        "n_closed_triples": n_closed,
        "n_violating_triples": n_viol,
        "intransitivity_rate": (n_viol / n_closed) if n_closed else float("nan"),
        "n_open_triples_excluded": n_open,
        "open_triple_fraction": (n_open / (n_open + n_closed)) if (n_open + n_closed) else float("nan"),
    }


def transitive_closure_blowup(event_id, cand_i, cand_j, is_dup):
    """How much does naive transitive closure over-merge?

    Reports, per event, the number of connected components induced by the
    duplicate edges and the size of the largest one. A giant component means
    closure has chained distinct particles together -- the Hassanzadeh failure
    mode. Compare against the number of duplicate edges: if components are far
    fewer than edges and one is huge, closure is unsafe at that threshold.
    """
    events = np.unique(event_id)
    sizes, n_comps = [], []
    for ev in events:
        m = (event_id == ev) & (is_dup.astype(bool))
        ii, jj = cand_i[m], cand_j[m]
        parent: dict[int, int] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in zip(ii, jj):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
        if not parent:
            continue
        comps: dict[int, int] = {}
        for x in parent:
            r = find(x)
            comps[r] = comps.get(r, 0) + 1
        n_comps.append(len(comps))
        sizes.append(max(comps.values()))
    if not sizes:
        return {}
    return {
        "mean_components_per_event": float(np.mean(n_comps)),
        "mean_largest_component": float(np.mean(sizes)),
        "max_largest_component": int(np.max(sizes)),
    }


# ----------------------------------------------------------------------------
def _self_test():
    print("=" * 74)
    print("SELF-TEST: intransitivity counting on hand-built graphs")
    print("=" * 74)

    # --- a single planted violation: a~b, b~c, a!~c, all three scored
    ev = np.array([0, 0, 0])
    ci = np.array([0, 1, 0])
    cj = np.array([1, 2, 2])
    dup = np.array([1, 1, 0])
    r = intransitivity_rate(ev, ci, cj, dup)
    print(f"\n  planted 1 violating triple:")
    print(f"    closed triples {r['n_closed_triples']}  violations "
          f"{r['n_violating_triples']}  rate {r['intransitivity_rate']:.3f}")
    assert r["n_closed_triples"] == 1 and r["n_violating_triples"] == 1
    assert r["intransitivity_rate"] == 1.0
    print("    OK")

    # --- a consistent triangle: all three duplicate
    dup2 = np.array([1, 1, 1])
    r2 = intransitivity_rate(ev, ci, cj, dup2)
    print(f"\n  consistent triangle (all three duplicate):")
    print(f"    closed {r2['n_closed_triples']}  violations "
          f"{r2['n_violating_triples']}  rate {r2['intransitivity_rate']:.3f}")
    assert r2["n_violating_triples"] == 0
    print("    OK")

    # --- open triple: (a,c) never scored, must be EXCLUDED not counted
    ev3 = np.array([0, 0])
    ci3 = np.array([0, 1])
    cj3 = np.array([1, 2])
    dup3 = np.array([1, 1])
    r3 = intransitivity_rate(ev3, ci3, cj3, dup3)
    print(f"\n  open triple ((a,c) outside the window, never scored):")
    print(f"    closed {r3['n_closed_triples']}  open/excluded "
          f"{r3['n_open_triples_excluded']}")
    assert r3["n_closed_triples"] == 0 and r3["n_open_triples_excluded"] == 1
    print("    OK -- 'never asked' is not counted as 'denied'")

    # --- realistic synthetic: a scorer with independent per-pair noise
    print("\n" + "=" * 74)
    print("HOW MUCH INTRANSITIVITY DOES AN INDEPENDENT PAIRWISE SCORER PRODUCE?")
    print("=" * 74)
    rng = np.random.default_rng(0)
    n_events, n_cand = 60, 40
    E, I, J, D = [], [], [], []
    for e in range(n_events):
        # ground-truth clusters of size 1-4 sharing a sim track
        labels = rng.integers(0, n_cand // 3, n_cand)
        for a in range(n_cand):
            for b in range(a + 1, n_cand):
                if rng.random() > 0.25:      # dR window keeps ~25% of pairs
                    continue
                true_dup = labels[a] == labels[b]
                # independent noise per pair: this is the mechanism
                flip = rng.random() < (0.10 if true_dup else 0.02)
                E.append(e); I.append(a); J.append(b)
                D.append(int(true_dup != flip))
    E, I, J, D = map(np.array, (E, I, J, D))

    truth = intransitivity_rate(E, I, J, D)
    print(f"\n  events {truth['n_events']}  duplicate edges "
          f"{truth['n_duplicate_edges']:,}")
    print(f"  closed triples {truth['n_closed_triples']:,}   violations "
          f"{truth['n_violating_triples']:,}")
    print(f"  INTRANSITIVITY RATE = {truth['intransitivity_rate']:.4f}")
    print(f"  open triples excluded: {truth['n_open_triples_excluded']:,} "
          f"({truth['open_triple_fraction']:.1%} of all triples)")

    bl = transitive_closure_blowup(E, I, J, D)
    print(f"\n  naive transitive closure:")
    print(f"    components/event {bl['mean_components_per_event']:.1f}   "
          f"mean largest {bl['mean_largest_component']:.1f}   "
          f"max largest {bl['max_largest_component']}")
    print(f"    (n_cand = {n_cand}; a largest component approaching that means")
    print("     closure has chained the whole event into one blob)")

    # --- sanity: a PERFECT scorer must be exactly transitive
    Dp = []
    rng2 = np.random.default_rng(0)
    E2, I2, J2 = [], [], []
    for e in range(n_events):
        labels = rng2.integers(0, n_cand // 3, n_cand)
        for a in range(n_cand):
            for b in range(a + 1, n_cand):
                if rng2.random() > 0.25:
                    continue
                E2.append(e); I2.append(a); J2.append(b)
                Dp.append(int(labels[a] == labels[b]))
    rp = intransitivity_rate(np.array(E2), np.array(I2), np.array(J2), np.array(Dp))
    print(f"\n  control -- a PERFECT scorer: rate = {rp['intransitivity_rate']:.6f}")
    assert rp["n_violating_triples"] == 0, "perfect labels must be transitive"
    print("    OK -- 0 by construction, so a nonzero rate on real data measures")
    print("    scorer inconsistency and nothing else.")

    print("\nall assertions passed")


if __name__ == "__main__":
    _self_test()
