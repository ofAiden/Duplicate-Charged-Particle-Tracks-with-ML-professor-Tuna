"""
Generate the LaTeX results tables directly from the run artifacts.

Numbers are never transcribed by hand: every figure in the paper is emitted from
results.json / arbiter_results.json / the pairs npz, so the draft cannot drift
from what the code actually produced.

    python paper/make_tables.py --results results --pairs pairs.npz \
        --out paper/tables.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fmt(x, n=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return r"\textemdash"
    return f"{x:.{n}f}"


def table_metrics(res):
    rows = []
    for k in ("euclidean", "mahalanobis", "cosine"):
        if k not in res:
            continue
        v = res[k]
        h = v["history"][-1]
        rows.append((
            k.capitalize(),
            fmt(v["auc_t5_balanced"]), fmt(v["auc_pls_balanced"]),
            fmt(h.get("rankme"), 2), f"{h.get('var_top3', float('nan'))*100:.1f}\\%",
        ))
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Metric comparison with each metric trained independently under its own
objective. RankMe is the effective rank of the 12-dimensional embedding
(\cref{{sec:collapse}}); var@3 is the variance captured by the leading three
principal components.}}
\label{{tab:metrics}}
\begin{{tabular}}{{lcccc}}
\toprule
Metric & AUC (T5--T5) & AUC (pLS--T5) & RankMe & var@3 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_operating(res, prior="0.05"):
    rows = []
    for k in ("euclidean", "mahalanobis", "cosine"):
        op = res.get(k, {}).get(f"t5_at_prior_{prior}")
        if not op:
            continue
        rows.append((
            k.capitalize(),
            fmt(op["auc_balanced"]),
            fmt(op["duplicate_recall"], 3),
            fmt(op.get("blocking_ceiling"), 4),
            fmt(op.get("duplicate_recall_end_to_end"), 3),
        ))
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Balanced-pair AUC against deployment-relevant performance at a
{float(prior)*100:.0f}\% duplicate prior and $99\%$ track efficiency. End-to-end
recall folds in the blocking ceiling of the $\Delta R^2<0.02$ window.}}
\label{{tab:operating}}
\begin{{tabular}}{{lcccc}}
\toprule
Metric & Balanced AUC & Recall (in-window) & Ceiling & Recall (end-to-end) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_arbiter(arb):
    order = [("embedding", "Embedding only"), ("raw", "Raw features"),
             ("both", "Embedding + raw")]
    rows = []
    for key, label in order:
        r = arb["by_input"].get(key)
        if not r:
            continue
        rows.append((label, fmt(r["accuracy"], 4),
                     fmt(r["accuracy_decisive_half"], 4), fmt(r["ranking_auc"], 4)))
    rows.append(("Collection priority (baseline)", "0.5000", "0.5000", "0.5000"))
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    ratio = arb.get("dup_separation_ratio", float("nan"))
    return rf"""\begin{{table}}[t]
\centering
\caption{{Arbitration accuracy by input representation. The collection-priority
rule carries no information for same-collection pairs and is a coin flip by
construction. The trained embedding separates duplicate pairs to
${ratio:.4f}$ of the embedding scale, which is why the embedding-only head
underperforms.}}
\label{{tab:arbiter}}
\begin{{tabular}}{{lccc}}
\toprule
Input & Accuracy & Accuracy (decisive half) & Ranking AUC \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def macros(res, pairs, arb):
    """\newcommand definitions so prose can cite numbers without transcription."""
    d = np.load(pairs)
    ql, qr, y = d["q_left_t5"], d["q_right_t5"], d["y_t5"]
    dup = y == 0
    gap = np.abs(ql[dup] - qr[dup])
    e = res["euclidean"]
    h = e["history"][-1]
    op = e["t5_at_prior_0.05"]
    m = {
        "NpairsTfive": f"{len(y):,}",
        "CeilingTfive": f"{float(d['blocking_ceiling_t5']):.4f}",
        "CeilingCross": f"{float(d['blocking_ceiling_pls']):.4f}",
        "RankMeEucl": f"{h.get('rankme', float('nan')):.2f}",
        "VarTopThree": f"{h.get('var_top3', float('nan'))*100:.1f}",
        "AucEucl": f"{e['auc_t5_balanced']:.4f}",
        "RecallDeploy": f"{op['duplicate_recall']:.3f}",
        "RecallEndToEnd": f"{op['duplicate_recall_end_to_end']:.3f}",
        "TieFraction": f"{(gap == 0).mean()*100:.1f}",
        "DecisiveFraction": f"{(gap > 1e-3).mean()*100:.1f}",
        "ArbAcc": f"{arb['by_input']['both']['accuracy']:.4f}",
        "ArbGain": f"{arb['by_input']['both']['accuracy'] - arb['by_input']['embedding']['accuracy']:+.4f}",
        "DupSep": f"{arb.get('dup_separation_ratio', float('nan')):.4f}",
        "Nevents": f"{int(np.unique(d['event_id_t5']).size)}",
    }
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pairs", default="pairs.npz")
    ap.add_argument("--out", default="paper/tables.tex")
    a = ap.parse_args()

    rd = Path(a.results)
    res = json.loads((rd / "results.json").read_text())
    arb_p = rd / "arbiter_results.json"
    arb = json.loads(arb_p.read_text()) if arb_p.exists() else None

    header = "% AUTO-GENERATED by paper/make_tables.py -- do not edit by hand."

    # macros go in the PREAMBLE; table environments must go in the BODY, so they
    # are emitted to separate files
    out = Path(a.out)
    macro_path = out.with_name("macros.tex")
    macro_path.write_text(
        header + "\n" + (macros(res, a.pairs, arb) if arb else "") + "\n")

    tables = [table_metrics(res), table_operating(res)]
    if arb:
        tables.append(table_arbiter(arb))
    out.write_text(header + "\n\n" + "\n\n".join(tables) + "\n")
    print(f"wrote {macro_path} and {out}")


if __name__ == "__main__":
    main()
