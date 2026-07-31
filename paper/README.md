# Paper draft

## Getting this into Overleaf

I cannot push to Overleaf directly — it needs your account credentials, and no
Overleaf integration is connected to this session. Two ways to get it there:

**Upload a zip (simplest).** From the repo root:

```bash
cd paper && zip -r ../lst-paper.zip main.tex macros.tex tables.tex && cd ..
```

Then in Overleaf: *New Project → Upload Project* and pick `lst-paper.zip`.

**Or connect via git.** Create a blank Overleaf project, use *Menu → Git* to get
its URL, then push this directory to it. That keeps `make_tables.py` able to
regenerate the numbers and lets you `git push` updates.

## Regenerating the numbers

**Never edit `macros.tex` or `tables.tex` by hand** — both are generated, and
hand edits will be silently overwritten and will let the draft drift from what
the code actually produced.

```bash
python paper/make_tables.py --results results --pairs pairs.npz --out paper/tables.tex
cd paper && tectonic main.tex
```

`macros.tex` holds `\newcommand` definitions used inline in the prose (so a
number quoted in a sentence and the same number in a table cannot disagree).
`tables.tex` holds the table environments and is `\input` in the document body.

## Reproducing the results

```bash
python run_real.py --events 500 --out pairs.npz        # ~40 min, needs data_file.root
python metric_study.py --pairs pairs.npz --epochs 60 --emb-dim 12 --out results
python train_arbiter.py --pairs pairs.npz --encoder results/euclidean.pt \
    --emb-dim 12 --epochs 40 --out results
```

`data_file.root` (3.2 GB) is gitignored; `run_real.py` expects it in the repo
root. The notebook downloads it from Google Drive.

## Status of the draft

Sections and claims are written against measurements that exist. Two things are
still open and are marked in the text:

- `\todo{}` in §Limitations: no end-to-end MTV efficiency / fake rate / duplicate
  rate versus the production baseline. That requires CMSSW integration and is the
  main thing a CMS referee will ask for.
- The intransitivity figures in §6 come from a densely-paired 6-event subset,
  not the main run, because the per-event pair cap starves the measurement. This
  is stated in the text but the subset is small and should be enlarged.

Claim scoping that must not be quietly widened, and why:

- The gauge-freedom result in §4 is **known** for linear metric learning
  (Weinberger & Saul 2009; Kulis 2013; Bellet et al. 2013). The draft says so.
  Do not restate it as novel.
- Learned duplicate arbitration in tracking is **already published** for ACTS
  (Allaire et al., EPJ Web Conf. 337, 01025 (2025) — no arXiv version). Phrases
  like "first learned arbitration" are not available.
- The blocking-recall-ceiling measurement appears to be unreported in HEP
  tracking, but that rests on keyword searches over a partially-readable corpus
  (CDS is behind bot protection). Treat as "we are not aware of" rather than
  "nobody has".
- ACAT 2025 proceedings were still unindexed as of 2026-07-29. The LST team's own
  write-up is the likeliest self-scoop — **check before submitting**.
