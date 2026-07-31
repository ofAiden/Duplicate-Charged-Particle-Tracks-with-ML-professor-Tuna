# Paper status — LST duplicate removal

Working document. Last updated 2026-07-29.

Every number below was verified numerically or against a primary source. Where
something is unverified it says so.

## The constraint that shapes everything

The project's core method is **already published by CMS and already merged into
CMSSW production**.

- **CERN-CMS-DP-2025-048** (Niendorf, 2025-08-22, ACAT 2025). Verified from
  `twiki.cern.ch/twiki/bin/view/CMSPublic/DP2025048`: T5 (30 features) and pLS
  (10 features) DNNs → two 32-unit Linear+ReLU hidden layers → shared **6-D**
  embedding, Euclidean distance into a contrastive loss, margin 1.0. Reports
  −50% duplicate rate for displaced tracks, ~40% in barrel, and an embedding
  overhead of 3.47→3.57 ms (~3% at 1 stream), 1.62→1.69 ms (~4% at 8 streams)
  on an NVIDIA L40.
- Merged commit `ab51b65ac1` (2025-06-04). `Params_T5::kEmbed = 6`,
  `t5embdnn::runEmbed`, `CrossCleanT5` with `dR2 < 0.02f` — the same constant as
  `DELTA_R2_CUT` in the notebooks. **No L2 normalisation**, so the production
  metric is raw Euclidean.
- **No arXiv or journal version exists** (INSPIRE report-number and title
  searches both return 0; author search for Niendorf shows nothing on embeddings
  or duplicate removal).

"We introduce learned embeddings for LST duplicate removal" is not available.

## Claim status

### C1 — global learned Mahalanobis is a gauge freedom — **DEMOTED to a note**

The mathematics is verified (`verify_gauge_freedom.py`): with `V = LᵀL` and an
encoder ending in a trainable linear layer, the model class {encoder ending in
W, metric V} equals {encoder ending in LW, metric I}. Planted exact solution
gives residual 1.6e-13; fitted from scratch reaches 0.00000.

But it does not support a paper claim:

- **Textbook.** Weinberger & Saul, JMLR 10:207–244 (2009) §2.3 states the
  `M = LᵀL` ↔ linear-transform equivalence outright; §2.1 states the rotational
  gauge. Also in Kulis, FnT ML 5(4) §2.2.1 and Bellet et al. arXiv:1306.6709 §3.
- **The framing is published.** Cain, arXiv:2603.06774 (2026-03-06), "Gauge
  Freedom and Metric Dependence in Neural Representation Spaces" — verified;
  uses that exact language for invertible linear reparameterisation of neural
  representations. It never touches metric learning or expressive power, but the
  terminology is taken.
- **The practice it would correct may not exist.** `all:"learnable Mahalanobis"`
  on arXiv returns **0 results**. An `abs:Mahalanobis AND abs:contrastive` sweep
  (25 most recent) finds only three papers learning a Mahalanobis
  parameterisation on a contrastive encoder, and none is "a global learned
  precision matrix on a learned embedding": arXiv:2607.18004 is the
  *pair-conditioned fix*, arXiv:2512.12069 and arXiv:2510.15977 are post-hoc
  detectors on activations. An earlier research pass claimed a "2025–26 cluster
  (KDD/AAAI)" doing this; that came from Scholar snippets it never opened and
  arXiv does not corroborate it. **Treat as unsupported.**

**Verified scope limits**, which matter if this is ever written up:
- A terminal ReLU breaks the absorption (rel. diff 0.42).
- L2 normalisation breaks it (rel. diff 0.955) — so it does not apply to
  L2-normalised HEP metric learning such as Chan, arXiv:2605.14131. It *does*
  apply to CMSSW, which is unnormalised.
- Weight decay: identical distances, 13.3× different Frobenius penalty. Same
  function class ≠ same optimisation problem. An ML-literate referee will raise
  this; address it explicitly.

**Disposition:** keep as a methods paragraph explaining why the project's
Mahalanobis arm produced a null. Do not sell it as a contribution.

**What survives as genuinely new:** the constructive half. A per-instance
precision is *not* absorbable — verified in `probabilistic_embedding.py`, control
absorbed to 0.00000 while per-instance MLS plateaus at 0.358 (flat across
25/50/100% of budget, so converged not underfit). Cost ~1.5k parameters.
Concurrent work: PAMD (arXiv:2607.18004, 2026-07-20) does pair-conditioned
Mahalanobis in visual RL against fixed ℓp baselines — must cite, nine days older.

### C2 — dimensional collapse explains CMS's own null — **SAFE, now the strongest claim**

DP-2025-048 reports, with a figure captioned *"Test loss at epoch 200 for
different embedding vector sizes, with the mean and standard deviation shown
over 5 random seeds"*, that **"Increasing the embedding dimension up to 32 shows
no improvement in test loss for either model."** That is a properly-seeded
output-dimension scan with error bars, distinct from the 32-unit hidden width
(twice verified; a mid-check doubt about conflation was wrong).

The project has the diagnostic that explains it, which the note never ran:
- RankMe effective rank **6.47/12** (T5) and **6.01/12** (cross); participation
  ratio 1.87/12; top-3 dims hold 94.5% of variance. From the committed
  `cov_t5.npy` / `cov_cross.npy`.
- RankMe implementation validated against known-rank inputs: 12.00 / 3.00 / 1.00.
- Embedding RMS norm 0.062 against a margin of 1.0 — an ~11× mismatch, so every
  pair sits far inside the margin and the repulsive term never saturates.

**Corpus-clean in HEP:** across all CMS DP notes, `Mahalanobis` → 0,
`"precision matrix"` → 0, `"metric learning"` → 0; `Euclidean` appears in exactly
one note ever (DP-2025-048). Across 1427 contributions at CTD 2025, ACAT 2025,
CHEP 2024 and CHEP 2026: zero hits for Mahalanobis, RankMe, effective rank,
dimensional collapse, precision matrix, antisymmetr.

### C3 — blocking recall ceiling — **WEAKENED twice, measurement survives**

Two independent hits:
- Allaire et al., **EPJ Web Conf. 337, 01025 (2025)** characterises its own
  DBSCAN grouping step (cluster purity, ε trade-off, efficiency vs ΔR to nearest
  track).
- **CMS-DP-2026-030** (2026-05-18) uses a geometrically motivated ΔR < 0.1
  attention mask on LST T3s.

Drop "unmeasured and unreported in HEP tracking." What survives: nobody measures
a *recall ceiling* for the pair-forming window — `recall`, `"false negative"` and
`"true duplicate"` all return 0 across the CMS DP corpus. Level mismatch also
favours us: DP-2026-030's mask is over T3 *tokens* for attention, not candidate
*pairs* for duplicate removal.

Implemented in `build_pairs.py`; self-test recovers a planted ceiling of exactly
0.500.

### C4 — learned asymmetric arbitration — **WEAKENED twice**

- Allaire et al., **EPJ Web Conf. 337, 01025 (2025)** — 4D DBSCAN over
  (η, φ, z₀, p_T) then a 3-layer MLP under margin ranking loss; *"for each
  cluster, only the highest score will be kept."* Published, ACTS-integrated,
  learned which-to-keep. **No arXiv version, which is why it was invisible.**
- **CMS-DP-2026-030** — object condensation's β charge is a learned per-object
  confidence and DBSCAN replaces arbitration outright.

"First learned arbitration in tracking" and "first learned duplicate resolution
in CMS" are both dead. Surviving deltas, all real:
- Antisymmetric *pairwise comparator* (`s(i,j) = g(e_i,e_j) − g(e_j,e_i)`,
  measured error exactly 0.0) versus score-then-argmax over a hand-coded cluster.
- Works without shared hits, on *typed heterogeneous* collections.
- Replaces the CMSSW `pT5 > pT3 > T5 > pLS` priority, which DP-2026-029 confirms
  is still in force: *"A final list of Track Candidates (TC) is created by
  collecting pT5s, pT3s, T5s, and unused-hit initial-iteration pixel seeds
  (pLS)."* `arbitration` appears in no CMS DP note since 2014; `priority` and
  `ranking` → 0 in the five-note scope.

### NEW — the T4 gap — **the strongest forward opportunity**

**CMS-DP-2026-029** (2026-05-18, "Extending the reconstruction of Phase-2
displaced tracks using LST") introduces a new **T4 / quadruplet** candidate type
and reports that adding T4s **increases** the duplicate rate by a few percent
versus vertex radius.

The DP-2025-048 embedding is trained only on T5-T5, T5-pT5 and pLS-T5. **T4s are
uncovered.** This is a CMS-documented, live, unaddressed problem with a clear
owner — a better position than re-litigating a metric comparison.

## REAL DATA (2026-07-30)

`data_file.root` was obtainable all along — it is a public Drive link, pulled
with `gdown` (3.2 GB, 500 events, 252 branches). Everything below is measured on
it, not synthetic.

**Feature builder validated:** `run_real.py` reproduces **3,630,781 T5
candidates**, exactly matching the count the notebook reports, so the feature
definitions are faithfully replicated.

**Blocking recall ceiling, full 500 events** — the first such measurement in HEP
tracking as far as the 2026-07 sweep could establish:

| pairing | true duplicate pairs | inside ΔR² < 0.02 | ceiling |
|---|---|---|---|
| T5–T5 | 29,132,826 | 28,994,422 | **0.9952** |
| pLS–T5 | 6,297,847 | 6,067,117 | **0.9634** |

The cross-collection window is the tighter constraint, consistent with pLS and
T5 candidates for one particle being further apart in η–φ than two T5s.

**`t5_pMatched` — the arbitration blocker, resolved.** Quantised to 11 values
over 3.63M candidates (median 1.0, std 0.115). On a 25-event subset, **55.5% of
duplicate pairs are exact ties** and **44.5% are decisive**. Enough to train on;
ties carry no signal and are excluded.

**Dimensional collapse confirmed on real data** (25-event run, corrected
harness): Euclidean RankMe **6.05/12**, var@3 **96.2%** — closely matching the
6.47/12 computed independently from the committed covariance files, and the
notebook's 95.8% var@3. **New finding: cosine resists collapse**, RankMe 10.35 vs
Euclidean 6.05.

**Balanced AUC overstates deployment badly:** AUC 0.9524 but duplicate recall at
a 5% prior and 99% track efficiency is **0.209** in-window, **0.207** end-to-end.

**Arbitration works:** 0.7025 accuracy (0.873 on the decisive half) vs 0.5 for
the collection-priority rule. The predicted information-destruction effect is
real — duplicates separated to **0.158** of the embedding scale, and raw features
add **+0.066** over the embedding alone.

**Intransitivity:** ground truth gives 14,655,360 closed triples and exactly
zero violations; the model is intransitive on **3.6%–16.9%** of closed triples.
Requires dense pairing — at the standard per-event cap, 98.6% of triples have an
unscored edge and the measurement is starved.

Full 500-event training and a 40-event dense run are in progress; the numbers
above from 25-event and 6-event subsets will be superseded.

## Verified evidence base

| Result | Check | Value |
|---|---|---|
| Gauge absorption (global) | planted analytic solution | 1.6e-13 |
| Gauge absorption (global) | fitted from scratch | 0.00000 |
| Per-instance NOT absorbable | same budget, converged | 0.358 |
| Terminal ReLU breaks it | rel. diff | 0.42 |
| L2 norm breaks it | rel. diff | 0.955 |
| RankMe correctness | known-rank inputs | 12.00 / 3.00 / 1.00 |
| Collapse (real data) | RankMe from `cov_t5.npy` | 6.47 / 12 |
| Mahalanobis scalar share | from `cov_t5.npy` | 94.9% |
| Blocking ceiling | planted 0.500 | 0.500 |
| Event-level CRC | 300 splits, α=0.01 | 0.00591 OK |
| Pooled-pair CRC (the old bug) | same splits | 0.01028 VIOLATED |
| Arbiter antisymmetry | by construction | exactly 0.0 |
| Set arbiter equivariance | random permutation | 3.8e-06 |
| Comparison head symmetry | by construction | exactly 0.0 |
| linear(diff²) = sq. Euclidean | normalised MSE | 0.000000 |
| linear(diff²) = diag Mahalanobis | normalised MSE | 0.000000 |
| linear(diff²) ≠ full Mahalanobis | normalised MSE | 0.483 |
| +outer terms = full Mahalanobis | normalised MSE | 0.000000 |
| MLP vs fitted quadratic, conjunctive | ΔAUC | **−0.0026** |
| MLP vs fitted quadratic, disjunctive | ΔAUC | **+0.2664** |
| Intransitivity, noisy pairwise scorer | closed triples | 0.391 |
| Intransitivity, perfect scorer | control | 0.000000 |
| Open (unclosable) triples | ΔR window | 75.6% |
| Track-destruction risk monotone | CRC precondition | asserted |
| Track destruction at t=0.5 | same-track merges only | 0.0003 |

All on synthetic data with known answers, except the four drawn from the
committed covariance files. **No result on real physics data yet** — blocked on a
cluster run.

## Open items

1. **Run on real data.** Nothing here is a physics result until it runs on the
   ROOT file.
2. **Is `t5_pMatched` decisive between duplicates?** If near-constant, the
   arbitration head has no label. Highest-leverage unknown.
3. **Check ACAT 2025 proceedings before submitting** — still not on INSPIRE
   (`cnum C25-09-08` → 0). The LST team's own write-up is the likeliest
   self-scoop.
4. **Two DP notes need a human eye:** CMS-DP-2026-011 (contains "duplicate
   removal" + LST + ranking) and CMS-DP-2026-114 (contains "embedding" + LST).
5. **CDS is behind proof-of-work bot protection**, so DP-note body text is
   partial and the phrase-level negatives above rest on a fulltext index with
   known font-encoding problems. Strong evidence, not absolute.

## Code

```
build_pairs.py              pairs + quality + blocking ceiling + candidate table
metric_study.py             3 metrics trained independently, RankMe, conformal
conformal_risk.py           event-level CRC, track-destruction risk, Mondrian
arbitration.py              antisymmetric pairwise + equivariant set heads
train_arbiter.py            arbiter training + information-destruction ablation
probabilistic_embedding.py  per-instance precision, PFE MLS, HIB soft loss
comparison_head.py          learnable symmetric comparison; containment measured
transitivity.py             intransitivity rate + transitive-closure blowup
diagnose_metric_degeneracy.py   the Mahalanobis null, from saved covariances
verify_gauge_freedom.py     the gauge result + its four scope limits
validate_conformal.py       coverage under distribution shift
export_pairs.py             notebook -> npz bridge
```

All 12 self-test on synthetic data with known answers. Run any of them directly.

## Two findings that are the same limitation seen twice

The ΔR < 0.02 window caps duplicate **recall** (blocking ceiling) *and* leaves
**75.6% of triples unclosable**, so global consistency cannot even be checked for
most of the graph. Reporting them together is stronger than either alone, and
both are measurements nobody in HEP tracking has published.
