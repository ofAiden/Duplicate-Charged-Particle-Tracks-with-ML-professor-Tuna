"""
Empirical validation of the split-conformal threshold used in metric_study.py.

The claim the paper will have to defend is:

    Choosing the threshold as the floor(alpha*(n+1))-th smallest distance among
    n genuine non-duplicate calibration pairs guarantees that a future genuine
    pair is wrongly removed with probability at most alpha -- with no
    assumption on the distance distribution, only exchangeability.

This script checks that (a) the guarantee holds across many random
calibration/test splits and across wildly different distance distributions, and
(b) it FAILS under distribution shift, which is the honest caveat that belongs
in the paper.

Run:  .venv/bin/python validate_conformal.py
"""

import numpy as np

from metric_study import conformal_threshold

RNG = np.random.default_rng(7)
N_TRIALS = 2000


def coverage_trial(sampler, n_calib, alpha, n_test=5000, shift=None):
    """One draw: calibrate on n_calib genuine pairs, measure the false-removal
    rate on fresh genuine pairs."""
    calib = sampler(n_calib)
    thr = conformal_threshold(calib, alpha)
    test = sampler(n_test) if shift is None else shift(n_test)
    return (test < thr).mean()


def report(name, sampler, alpha, n_calib, shift=None, shift_name=""):
    rates = np.array([coverage_trial(sampler, n_calib, alpha, shift=shift)
                      for _ in range(N_TRIALS)])
    tag = f"{name}{(' -> ' + shift_name) if shift_name else ''}"
    ok = rates.mean() <= alpha * 1.05
    print(f"  {tag:<38} mean={rates.mean():.5f}  "
          f"p95={np.quantile(rates, 0.95):.5f}  target<={alpha:.4f}  "
          f"{'OK' if ok else '** VIOLATED **'}")
    return rates.mean()


def main():
    alpha = 0.005
    n_calib = 20000

    print(f"Split-conformal false-removal control | alpha={alpha}  "
          f"n_calib={n_calib}  trials={N_TRIALS}\n")

    print("1. Guarantee holds regardless of the distance distribution")
    print("   (exchangeable calibration and test):")
    dists = {
        "half-normal (Euclidean-like)": lambda n: np.abs(RNG.standard_normal(n)),
        "chi (Mahalanobis-like, d=12)": lambda n: np.sqrt((RNG.standard_normal((n, 12)) ** 2).sum(1)),
        "heavy-tailed lognormal": lambda n: RNG.lognormal(0, 1.5, n),
        "bimodal": lambda n: np.where(RNG.random(n) < 0.3,
                                      RNG.normal(0.5, 0.1, n), RNG.normal(3, 0.5, n)),
        "near-collapsed (tiny scale)": lambda n: np.abs(RNG.standard_normal(n)) * 0.06,
    }
    for name, s in dists.items():
        report(name, s, alpha, n_calib)

    print("\n2. Calibration-set size controls the variance, not the mean:")
    base = dists["chi (Mahalanobis-like, d=12)"]
    for n in (1000, 5000, 20000, 100000):
        rates = np.array([coverage_trial(base, n, alpha) for _ in range(N_TRIALS)])
        print(f"  n_calib={n:>7}  mean={rates.mean():.5f}  "
              f"sd={rates.std():.5f}  p95={np.quantile(rates, 0.95):.5f}")

    print("\n3. THE CAVEAT -- exchangeability is required. Under distribution")
    print("   shift the guarantee is void:")
    shifts = {
        "distances shrink 20% (higher PU)": lambda n: base(n) * 0.8,
        "distances shrink 40%": lambda n: base(n) * 0.6,
        "5% contamination at small d": lambda n: np.where(
            RNG.random(n) < 0.05, RNG.random(n) * 0.5, base(n)),
    }
    for sname, sh in shifts.items():
        report("chi d=12", base, alpha, n_calib, shift=sh, shift_name=sname)

    print("\nConclusion for the paper: the threshold carries a finite-sample,")
    print("distribution-free guarantee on efficiency loss under exchangeability,")
    print("which is exactly what a trigger working point needs -- but it must be")
    print("recalibrated per pileup regime, and that caveat must be stated.")


if __name__ == "__main__":
    main()
