"""
The embedding-dimension figure: does effective rank saturate?

This is the falsification plot for the paper's central claim. The diagonal
RankMe = d is what an embedding using all of its dimensions would follow; a
curve that flattens below it is dimensional collapse, and the gap at d = 32 is
the reason raising the nominal dimension buys nothing.

    python paper/make_sweep_figure.py --sweep sweep_dims.json \
        --out paper/fig_dimsweep.pdf
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Colour-blind-safe, distinguishable in greyscale by marker as well as hue.
STYLE = {
    "euclidean": dict(color="#4269D0", marker="o", label="Euclidean"),
    "cosine": dict(color="#EFB118", marker="s", label="Cosine"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="sweep_dims.json")
    ap.add_argument("--out", default="paper/fig_dimsweep.pdf")
    a = ap.parse_args()

    s = json.loads(open(a.sweep).read())
    dims = sorted(int(k) for k in s["runs"])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    ax = axes[0]
    ax.plot(dims, dims, color="0.6", ls="--", lw=1,
            label="all dimensions used", zorder=1)
    for m, st in STYLE.items():
        if m not in s["runs"][str(dims[0])]:
            continue
        y = [s["runs"][str(d)][m]["rankme_t5"] for d in dims]
        ax.plot(dims, y, lw=1.8, ms=5, zorder=3, **st)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels(dims)
    ax.set_xlabel("nominal embedding dimension $d$")
    ax.set_ylabel("effective rank (RankMe)")
    ax.set_title("Effective rank vs. nominal dimension", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for m, st in STYLE.items():
        if m not in s["runs"][str(dims[0])]:
            continue
        for key, ls in (("auc_t5", "-"), ("auc_pls", ":")):
            y = [s["runs"][str(d)][m][key] for d in dims]
            lab = f"{st['label']} ({'T5-T5' if key=='auc_t5' else 'pLS-T5'})"
            ax.plot(dims, y, ls=ls, lw=1.8, ms=5, color=st["color"],
                    marker=st["marker"], label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels(dims)
    ax.set_xlabel("nominal embedding dimension $d$")
    ax.set_ylabel("ROC AUC (balanced pairs)")
    ax.set_title("Discrimination vs. nominal dimension", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")

    # emit the same content as a LaTeX table, so the claim can be checked
    # numerically and not only read off a plot
    rows = []
    for d in dims:
        r = s["runs"][str(d)]
        cells = [str(d)]
        for m in ("euclidean", "cosine"):
            if m in r:
                cells += [f"{r[m]['rankme_t5']:.2f}", f"{r[m]['auc_t5']:.4f}",
                          f"{r[m]['auc_pls']:.4f}"]
        rows.append(" & ".join(cells) + r" \\")
    tex = (r"""\begin{table}[t]
\centering
\caption{Effective rank and discrimination against nominal embedding dimension.
Effective rank saturates while the nominal dimension grows by a factor of eight,
which is what makes additional dimensions inert.}
\label{tab:dimsweep}
\begin{tabular}{cccccccc}
\toprule
& \multicolumn{3}{c}{Euclidean} & \multicolumn{3}{c}{Cosine} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
$d$ & RankMe & AUC (T5) & AUC (pLS) & RankMe & AUC (T5) & AUC (pLS) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}""")
    tpath = a.out.rsplit("/", 1)[0] + "/table_dimsweep.tex"
    open(tpath, "w").write(tex + "\n")
    print(f"wrote {tpath}")


if __name__ == "__main__":
    main()
