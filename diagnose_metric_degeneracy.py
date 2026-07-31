"""
Diagnostics on the saved covariance matrices from updated_fixed-3.ipynb.

Answers two questions that decide whether the Euclidean/Mahalanobis/Cosine
comparison in the notebook is a valid experiment:

  Q1  Why are the T5-T5 Euclidean and Mahalanobis AUCs identical to 4 d.p.
      (0.9392 vs 0.9391)?
  Q2  How collapsed is the embedding, measured properly (effective rank)?

Run:  .venv/bin/python diagnose_metric_degeneracy.py
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

RIDGE = 1e-3  # the value hard-coded in MahalanobisMetric._estimate_v
RNG = np.random.default_rng(0)


def effective_rank(eigvals):
    """RankMe (arXiv:2210.02885): Shannon entropy of the normalised singular
    value spectrum, exponentiated.  Singular values of the embedding matrix
    scale as sqrt(eigenvalue) of the covariance."""
    sv = np.sqrt(np.clip(eigvals, 0, None))
    p = sv / sv.sum() + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigvals):
    """(sum l)^2 / sum(l^2).  A second, independent collapse measure."""
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


def analyse(name, cov):
    cov = (cov + cov.T) / 2
    d = cov.shape[0]
    lam = np.linalg.eigvalsh(cov)[::-1]  # descending
    V = np.linalg.inv(cov + RIDGE * np.eye(d))

    print(f"\n{'=' * 72}\n{name}   (embedding dim = {d})\n{'=' * 72}")

    print("\n-- collapse --")
    print(f"  variance explained by top 1/2/3 dims : "
          f"{lam[0]/lam.sum():.1%} / {lam[:2].sum()/lam.sum():.1%} / {lam[:3].sum()/lam.sum():.1%}")
    print(f"  RankMe effective rank                : {effective_rank(lam):.2f}  / {d}")
    print(f"  participation ratio                  : {participation_ratio(lam):.2f}  / {d}")
    print(f"  eigenvalue range                     : {lam[-1]:.3e} .. {lam[0]:.3e}")
    print(f"  total variance (trace)               : {lam.sum():.4e}")

    print("\n-- why V degenerates --")
    n_below = int((lam < RIDGE).sum())
    print(f"  ridge added by the code              : {RIDGE:.0e}")
    print(f"  eigenvalues SMALLER than the ridge   : {n_below}/{d}")
    print(f"  -> in {n_below} of {d} directions the ridge dominates the data,")
    print(f"     so V = inv(cov + ridge*I) -> (1/ridge)*I = {1/RIDGE:.0f}*I there.")
    Vk = np.mean(np.diag(V))
    rel = np.linalg.norm(V - Vk * np.eye(d)) / np.linalg.norm(V)
    print(f"  ||V - kI||_F / ||V||_F               : {rel:.4f}   (0 = exactly scalar)")

    print("\n-- consequence for the metric comparison --")
    # Draw embeddings with this covariance, form random pairs, compare the two
    # distance rankings.  ROC-AUC depends on the distance only through its
    # RANKING of pairs, so rank correlation upper-bounds how different the two
    # ROC curves can possibly be.
    n = 40000
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(d))
    diff = (RNG.standard_normal((n, d)) - RNG.standard_normal((n, d))) @ L.T
    d_eucl = np.sqrt((diff ** 2).sum(1))
    d_mahal = np.sqrt(np.einsum("ij,jk,ik->i", diff, V, diff))
    rho = spearmanr(d_eucl, d_mahal).statistic
    print(f"  Spearman rank corr(Euclidean, Mahalanobis) : {rho:.4f}")
    print("  ROC-AUC is a function of the pair RANKING only, so the closer this")
    print("  is to 1 the less room there is for the two AUCs to differ at all.")

    # How much of V's action is a pure rescaling?  Split V into its scalar part
    # and the remainder, and report the share of "work" done by each.
    Vk = np.mean(np.diag(V))
    scalar_share = Vk ** 2 * d / (np.linalg.norm(V) ** 2)
    print(f"  fraction of ||V||_F^2 that is pure scaling : {scalar_share:.1%}")

    return lam


def scale_vs_margin(lam, margin=1.0):
    """The contrastive margin only does work if it is comparable to the scale
    the embedding actually occupies."""
    rms_norm = np.sqrt(lam.sum())          # E||e||  for zero-mean embeddings
    typical_pair_dist = np.sqrt(2 * lam.sum())  # E||e_i - e_j|| for independent e
    print(f"  RMS embedding norm                   : {rms_norm:.4f}")
    print(f"  typical independent-pair distance    : {typical_pair_dist:.4f}")
    print(f"  contrastive margin used in the code  : {margin:.1f}")
    print(f"  margin / typical pair distance       : {margin / typical_pair_dist:.1f}x")


def main():
    lam_t5 = analyse("cov_t5.npy   (T5-T5, similar pairs)", np.load("cov_t5.npy").astype(np.float64))
    lam_cr = analyse("cov_cross.npy (pLS-T5, similar pairs)", np.load("cov_cross.npy").astype(np.float64))

    print(f"\n{'=' * 72}\nWHAT THE RIDGE SHOULD BE\n{'=' * 72}")
    print("  The code adds a FIXED ridge of 1e-3, but the right ridge is relative")
    print("  to the scale of the covariance it is regularising.\n")
    for name, lam in [("t5", lam_t5), ("cross", lam_cr)]:
        mean_eig = lam.mean()
        print(f"  {name:5s}: mean eig {mean_eig:.3e} | median eig {np.median(lam):.3e} | min eig {lam.min():.3e}")
        print(f"         fixed ridge 1e-3 is {1e-3/mean_eig:>6.1f}x the mean eigenvalue,")
        print(f"                            {1e-3/np.median(lam):>6.1f}x the median,")
        print(f"                            {1e-3/lam.min():>6.1f}x the smallest.")
        print(f"         scale-free alternative: ridge = 1e-3 * trace/d = {1e-3*mean_eig:.3e}\n")

    print(f"{'=' * 72}\nEMBEDDING SCALE vs CONTRASTIVE MARGIN\n{'=' * 72}")
    print("\n  T5-T5:")
    scale_vs_margin(lam_t5)
    print("\n  pLS-T5:")
    scale_vs_margin(lam_cr)
    print("\n  The notebook reports observed Euclidean distances in [0.0010, 0.2867],")
    print("  consistent with the above. Every pair therefore sits far inside a")
    print("  margin of 1.0, so the repulsive term (margin - d)^2 is active for")
    print("  ~100% of dissimilar pairs and never saturates.")


if __name__ == "__main__":
    main()
