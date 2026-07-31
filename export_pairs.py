"""
Export the generated pairs from the notebook into the npz that metric_study.py
consumes, so the ~20 minutes of pair generation is paid once.

Paste EXPORT_SNIPPET into updated_fixed-3.ipynb as a new cell immediately after
the pLS-T5 pair-generation cell (cell 6), run it once, then on the cluster:

    python metric_study.py --pairs pairs.npz --epochs 60 --out results/

Note on what gets exported: metric_study.py does its own train/test split, so
this writes the FULL pair arrays, not the notebook's pre-split ones. That also
avoids inheriting the notebook's MAX_TRAIN_PAIRS=50000 subsample, which threw
away 95% of the 1M generated pairs.
"""

EXPORT_SNIPPET = '''
# ── Export pairs for metric_study.py ─────────────────────────────────────────
import numpy as np

# Full T5-T5 arrays, before any train/test split or subsampling.
# `X_left`, `X_right`, `y` and `weights_t5` are the outputs of cell 5.
_y_t5 = y if "y_t5" not in dir() else y_t5

np.savez_compressed(
    "pairs.npz",
    X_left    = np.asarray(X_left,    dtype=np.float32),
    X_right   = np.asarray(X_right,   dtype=np.float32),
    y_t5      = np.asarray(_y_t5,     dtype=np.float32),
    w_t5      = np.asarray(weights_t5, dtype=np.float32),
    # pLS-T5 arrays from cell 6
    X_pls     = np.asarray(pls_feats, dtype=np.float32),
    X_t5cross = np.asarray(t5_feats,  dtype=np.float32),
    y_pls     = np.asarray(y_pls,     dtype=np.float32),
    w_pls     = np.asarray(w_pls,     dtype=np.float32),
)

_d = np.load("pairs.npz")
print("wrote pairs.npz")
for k in _d.files:
    print(f"  {k:<10} {_d[k].shape}  {_d[k].dtype}")
print(f"  T5-T5  duplicate fraction: {(_d['y_t5'] == 0).mean():.3f}")
print(f"  pLS-T5 duplicate fraction: {(_d['y_pls'] == 0).mean():.3f}")
'''


def verify(path: str = "pairs.npz") -> None:
    """Check an exported npz has everything metric_study.py needs."""
    import numpy as np

    required = {
        "X_left": 30, "X_right": 30, "y_t5": None, "w_t5": None,
        "X_pls": 10, "X_t5cross": 30, "y_pls": None, "w_pls": None,
    }
    d = np.load(path)
    missing = [k for k in required if k not in d.files]
    if missing:
        raise SystemExit(f"missing arrays: {missing}")

    print(f"{path}: OK")
    for k, want_dim in required.items():
        a = d[k]
        note = ""
        if want_dim is not None and a.ndim == 2 and a.shape[1] != want_dim:
            note = f"  ** expected {want_dim} features, got {a.shape[1]} **"
        print(f"  {k:<10} {str(a.shape):<18} {a.dtype}{note}")

    n_t5 = len(d["y_t5"])
    n_pls = len(d["y_pls"])
    for name, arrs in [("T5-T5", ["X_left", "X_right", "y_t5", "w_t5"]),
                       ("pLS-T5", ["X_pls", "X_t5cross", "y_pls", "w_pls"])]:
        lens = {len(d[a]) for a in arrs}
        if len(lens) != 1:
            raise SystemExit(f"{name} arrays have mismatched lengths: {lens}")

    print(f"\n  T5-T5  {n_t5:>9,} pairs, duplicate fraction {(d['y_t5'] == 0).mean():.3f}")
    print(f"  pLS-T5 {n_pls:>9,} pairs, duplicate fraction {(d['y_pls'] == 0).mean():.3f}")
    print("\n  Reminder: y=0 means DUPLICATE, y=1 means not. metric_study.py"
          "\n  assumes that convention throughout.")

    for k in ("X_left", "X_right", "X_pls", "X_t5cross"):
        a = d[k]
        bad = ~np.isfinite(a)
        if bad.any():
            print(f"  ** {k}: {bad.any(axis=1).sum()} rows contain NaN/Inf **")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify(sys.argv[2] if len(sys.argv) > 2 else "pairs.npz")
    else:
        print(__doc__)
        print("Cell to paste into the notebook:")
        print("=" * 70)
        print(EXPORT_SNIPPET)
