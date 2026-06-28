# CLADE — DP Beta-Binomial VI for subclonal reconstruction

A self-contained, **PyClone-VI-style** pipeline that reconstructs the subclonal
architecture of a tumour from a single bulk DNA-sequencing sample, and scores its
predictions with the **SMC-Het / DREAM Somatic Mutation Calling — Tumour Heterogeneity**
metrics (sub-challenges 1A, 1B, 1C, 2A, 2B).

The clustering engine is a **truncated Dirichlet-Process mixture of Beta-Binomial
emissions**, with each cluster's cellular fraction represented as a categorical over a
fixed CCF grid (the PyClone-VI tractability trick), fit by **mean-field variational
inference (CAVI)**. It is deterministic, depends only on `numpy`/`scipy`, and runs in
**minutes for all 51 tumours** (vs hours for ABC/SMC samplers).

> **In one line:** infer *how many* tumour subclones there are, *what fraction of cells*
> each occupies, and *which mutation belongs to which* — directly from read counts.

---

## Table of contents

- [Highlights](#highlights)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Data](#data)
- [Outputs](#outputs)
- [Scoring harness](#scoring-harness-self-contained)
- [How the algorithm works (in detail)](#how-the-algorithm-works-in-detail)
- [References](#references)

---

## Highlights

- **Nonparametric** cluster number: a truncated DP (stick-breaking) *infers* the number
  of subclones instead of selecting it with a separate criterion.
- **Per-mutation Beta-Binomial** likelihood → native **soft** assignment and read-count
  overdispersion handling. This is the source of the engine's main strength (challenge 2A).
- **Neutral 1/f tail** component (MOBSTER-style) absorbs the diffuse neutral mutations so
  the Beta clusters stop over-splitting the tail.
- **Purity is inferred** from the data by maximising the variational ELBO (sub-challenge 1A
  comes out as a by-product, not a handed-in input).
- **Principled cluster-number control** via ICL-guided agglomerative merging (Baudry 2010),
  replacing hand-tuned merge thresholds; its entropy term is tempered (`icl_ent_w`) so sparse
  real subclones survive — lifting 1B past the old ~0.79 ceiling to ~0.81.
- **Fully self-contained**: input data, ground truth, and the SMC-Het scorer (ported to
  Python 3) are all vendored into this repository — no Python 2 required.

## Results

All 51 tumours, identical SMC-Het scoring harness, head-to-head against the DECODE SFS/ABC
engine (`DECODE_21`). Higher is better; full discussion in [`RESULTS.md`](RESULTS.md).

| Sub-challenge      | CLADE   | DECODE_21 (SFS/ABC) |
|--------------------|:-------:|:-------------------:|
| 2A (assignment)    | 0.600   | 0.567               |
| 1A (purity)        | 0.962   | 0.961               |
| 1B (cluster count) | 0.808   | 0.833               |
| 1C (prevalence)    | 0.937   | 0.942               |
| runtime            | minutes | hours               |

**Takeaway.** CLADE is a strictly better *assignment* engine and far faster. Its old
*cluster-count* (1B) ceiling of ~0.79 was lifted to **0.808** by tempering the ICL entropy term
(`icl_ent_w=0.25`) — the neutral tail already guards against over-split, so the entropy no longer
needs to, and stops merging away sparse real subclones. The gain to 1B came with **no cost** to
2A/1C (both slightly improved). See [`RESULTS.md`](RESULTS.md) for the full investigation.

## Repository layout

```
CLADE/
├── config.py                 # all paths & model hyper-parameters (PATHS, PARAMS)
├── run_cluster.sbatch        # SLURM array launcher for step 1 (1 sample / task)
├── model/
│   └── dpbb_vi.py            # ★ the engine: DP Beta-Binomial mixture + CAVI + selection
├── steps/
│   ├── 01_cluster.py         # step 1 — per-sample VI clustering        -> results/*.npz
│   ├── 02_scoring_files.py   # step 2 — build SMC-Het estimate files     -> scoring/pred_soft/
│   └── 03_score.py           # step 3 — run the scorer & aggregate       -> scoring/scores/
├── smc_het_eval/             # vendored SMC-Het scorer (Python 3, 4 files)
├── DREAM_data/               # input: per-sample mutation tables (51 CSVs)
├── scoring/
│   ├── truth/<sid>/          # ground truth per sample (the *.truth.2B.gz are gitignored)
│   ├── pred_soft/<sid>/      # generated estimate files (gitignored)
│   └── scores/               # per-sample scores (gitignored) + scores_summary.csv
├── results/                  # generated *_dpbbvi.npz clustering outputs (gitignored)
└── logs/                     # run logs (gitignored)
```

## Requirements

- **Python 3** with `numpy` and `scipy` — for everything: the engine, all steps, and the
  scorer. No Python 2 is needed (the vendored SMC-Het scorer was ported to Python 3).
- *(optional)* a SLURM cluster to run step 1 as an array job over the 51 samples.

No installation step is needed — clone the repo and run the scripts from its root.

## Quickstart

```bash
# 1) CLUSTER — per-sample VI clustering -> results/<sid>_dpbbvi.npz
#    all 51 samples as a SLURM array (1 sample / task):
sbatch run_cluster.sbatch
#    or locally, one sample at a time:
SLURM_ARRAY_TASK_ID=1 python3 steps/01_cluster.py
#    or locally, all samples in series:
python3 steps/01_cluster.py

# 2) ESTIMATE FILES — SMC-Het 1A/1B/1C/2A aligned to the scoring VCF
python3 steps/02_scoring_files.py
#    add the (disk-heavy) 2B co-clustering matrices:
python3 steps/02_scoring_files.py --with-2b --max-2b-n 20000

# 3) SCORE — run the vendored scorer and aggregate to scores_summary.csv
python3 steps/03_score.py soft --challenges 1A 1B 1C 2A
```

The final results land in [`scoring/scores/scores_summary.csv`](scoring/scores/scores_summary.csv)
(one row per sample, one column per sub-challenge).

## Configuration

All knobs live in [`config.py`](config.py) (`PARAMS`). The defaults reproduce the numbers in
[Results](#results).

| Parameter           | Default       | Meaning |
|---------------------|---------------|---------|
| `n_grid`            | `100`         | CCF grid resolution (number of bins on (0,1]). |
| `K_max`             | `12`          | DP truncation: maximum number of mixture components. |
| `alpha`             | `1.0`         | DP concentration (stick-breaking `Beta(1, alpha)`). |
| `precision`         | `200.0`       | Beta-Binomial precision (overdispersion). `None` → Binomial (over-splits). |
| `n_restarts`        | `10`          | VI restarts (k-means++ seeded); keep the best-ELBO fit. |
| `max_iter`, `tol`   | `400`, `1e-5` | CAVI iteration cap and relative-ELBO convergence tolerance. |
| `use_tail`          | `True`        | Add the fixed neutral power-law (1/f) component. |
| `tail_exponent`     | `2.0`         | Exponent of the neutral tail (Williams neutral evolution → 2.0). |
| `tail_min_ccf`      | `0.03`        | Lower CCF support of the tail (detection floor). |
| `purity_mode`       | `battenberg`  | Purity seed: `battenberg` (from `cellularity_ploidy.txt`) or `estimate` (from data). |
| `purity_search`     | `True`        | Refine purity by maximising the ELBO over a grid (this *infers* 1A). |
| `purity_span`/`steps`| `0.12`/`7`   | ± range and resolution of the purity grid search. |
| `selection`         | `icl`         | Cluster-number control: `icl` (principled merging) or `merge` (fixed-gap baseline). |
| `icl_ent_w`         | `0.25`        | Weight of the ICL entropy term; `<1` rescues sparse subclones (best 1B). See §10. |
| `mass_floor_frac`   | `0.005`       | Drop DP components below this fraction of mutations. |
| `merge_ccf`         | `0.12`        | CCF-gap threshold (only used when `selection="merge"`). |

## Data

Inputs are vendored into the repo — no external symlinks are needed.

**Input mutation tables** — `DREAM_data/<sid>_mutation_table_with_multiplicity.csv`, one row
per somatic SNV:

| Column        | Meaning |
|---------------|---------|
| `Chromosome`, `Position` | genomic locus (used as the `chrom_pos` key for VCF alignment) |
| `major_cn`, `minor_cn`   | allele-specific copy number (total CN = major + minor) |
| `Ref_count`, `Alt_count` | reference / variant supporting reads (depth = ref + alt) |
| `multiplicity`           | number of tumour chromosome copies carrying the mutation |

**Ground truth** — `scoring/truth/<sid>/` contains `*.truth.{1A,1B,1C,2A}.txt`,
`*.battenberg.txt`, `*.mutect.vcf`, and `*.truth.scoring_vcf.vcf` (the VCF that fixes the
mutation ordering for scoring).

**Cohort** — 51 tumours: 25 `P*` (real DECODE tumours), 9 `S*`, 17 `T*` (simulated).

**Not committed** (see [`.gitignore`](.gitignore)): the 2B co-clustering ground truth
`scoring/truth/**/*.truth.2B.gz` (up to ~2 GB/file — over GitHub's 100 MB/file limit) is kept
on disk locally but not published; provide it separately to score challenge 2B. Generated
outputs (`results/`, `scoring/pred_soft/`, `scoring/scores/` incl. `scores_summary.csv`, `logs/`)
are gitignored — `results/` and `scoring/scores/` keep an empty `.gitkeep` so the layout exists
on a fresh clone.

## Outputs

- **`results/<sid>_dpbbvi.npz`** — per-sample clustering: `chrom`, `pos`, `hard` (per-mutation
  cluster id), `soft` (N×K responsibilities), `ccf` (per-cluster CCF), `counts`, `purity`,
  `n_clusters`, `elbo`.
- **`scoring/pred_soft/<sid>/<sid>_estimate.{1A,1B,1C,2A}.txt`** (+ `2B.txt.gz`) — the SMC-Het
  prediction files (see [step 2](#7-from-clusters-to-smc-het-estimate-files-step-2)).
- **`scoring/scores/scores_summary.csv`** — the aggregated scores, one row per sample.

## Scoring harness (self-contained)

The SMC-Het scorer is **vendored** into this repo at [`smc_het_eval/`](smc_het_eval/) (four
files: `SMCScoring.py` + `scoring_harness_optimized.py` + `permutations.py` + `__init__.py` —
the exact import closure needed for challenges 1A/1B/1C/2A/2B). It was **ported to Python 3**
(from the original Python 2, via `2to3`) and is run as a subprocess by
[`steps/03_score.py`](steps/03_score.py) using the same Python 3 interpreter — no Python 2 needed.

---

# How the algorithm works (detailed)

This section explains the model and every computational step precisely, with pointers to the
code. The mathematics matches [`model/dpbb_vi.py`](model/dpbb_vi.py) line for line.

## 1. The problem

A bulk tumour sample is a mixture of normal cells and one or more tumour *subclones*. Each
somatic mutation `n` is observed through `b_n` variant reads out of depth `d_n`. The observed
**variant allele frequency** `b_n/d_n` depends on three things: the fraction of cells carrying
the mutation (its **cancer cell fraction**, CCF), the local copy number, and the sample
**purity** `ρ` (tumour-cell fraction). Subclonal reconstruction inverts this: group
mutations that share a CCF (a *clone/subclone*), estimate each group's CCF, and recover the
purity. The DREAM SMC-Het challenge scores exactly these quantities:

| Sub-challenge | Question | This pipeline's answer |
|---|---|---|
| **1A** | sample purity | `purity` (inferred by ELBO) |
| **1B** | number of clusters | `n_clusters` |
| **1C** | clusters: size + cellular prevalence | per-cluster `counts`, `ccf × purity` |
| **2A** | hard per-mutation assignment | `hard` |
| **2B** | soft co-clustering matrix | `soft @ soft.T` |

## 2. From read counts to expected VAF

For a mutation `n` with total copy number `c_n` and multiplicity `m_n`, in a sample of purity
`ρ`, a cluster whose cancer cell fraction is `φ` produces an **expected VAF**

```
ξ_n(φ) = ρ·m_n·φ / ( ρ·c_n + 2·(1−ρ) )
```

The denominator is the average ploidy at the locus (tumour copies weighted by `ρ`, two
normal copies weighted by `1−ρ`); the numerator is the number of mutated copies. This is
[`_expected_vaf`](model/dpbb_vi.py); inverting it gives the per-mutation MLE CCF used at
initialisation. CCF `φ ∈ (0,1]`, with `φ = 1` meaning *clonal* (present in every
tumour cell).

## 3. The generative model

A **truncated Dirichlet Process** (stick-breaking, Sethuraman 1994) mixture over the CCF grid
`{ (g−½)/G : g = 1..G }` (here `G = n_grid`):

```
v_k  ~  Beta(1, α),   π_k = v_k · ∏_{j<k} (1 − v_j),   k = 1..K_max
φ_k  ~  Categorical over the CCF grid
z_n  ~  Categorical(π)
b_n  ~  Binomial( d_n, ξ_n(φ_{z_n}) )   or   Beta-Binomial( d_n, ξ_n, s )
```

Two ideas make this tractable:

1. **CCF grid** (PyClone-VI, Gillis & Roth 2020). Representing each `φ_k` as a categorical
   over a fixed grid turns the non-conjugate Beta-Binomial emission into something with
   closed-form coordinate-ascent updates. The per-mutation, per-grid **emission
   log-likelihood**

   ```
   LL[n,g] = log p( b_n | d_n, ξ_n(g) )
   ```

   is precomputed once as an `N×G` matrix ([`_emission_loglik`](model/dpbb_vi.py)) and
   *shared by every cluster* — a cluster only differs through its grid posterior over `g`.
   Binomial if `precision=None`, Beta-Binomial with precision `s` otherwise.

2. **Truncated DP**: at most `K_max` components, but the stick-breaking prior drives unused
   components' weights to ≈0, so the *effective* number of clusters is learned, not fixed.

## 4. Mean-field variational inference (CAVI)

We approximate the posterior by a factorised distribution

```
q(z, v, φ) = ∏_n q(z_n) · ∏_k q(v_k) · ∏_k q(φ_k)
```

and maximise the **ELBO** (evidence lower bound) by coordinate ascent ([`_fit_once`](model/dpbb_vi.py)).
Each sweep updates, in closed form:

- **Cluster CCF** `q(φ_k)` — a categorical over the grid:
  ```
  log Phi_k  ∝  log(1/G)  +  Σ_n r_nk · LL[n,:]
  ```
  i.e. each cluster's CCF posterior is the responsibility-weighted sum of the mutation
  likelihoods (`logPhi = log_prior_phi + r.T @ LL`, then softmax over the grid).

- **Stick-breaking weights** `q(v_k) = Beta(a_k, b_k)` with
  `a_k = 1 + N_k` and `b_k = α + Σ_{j>k} N_j`, where `N_k = Σ_n r_nk`.
  The expected log-weights `E[log π_k]` follow from digamma identities
  ([`_stick_expectations`](model/dpbb_vi.py)).

- **Responsibilities** `q(z_n)`:
  ```
  r_nk  ∝  exp( E[log π_k] + E_{q(φ_k)}[ log p(b_n | φ_k) ] )
  ```
  where the Beta-cluster evidence `E_{q(φ_k)}[log p] = Σ_g Phi_kg · LL[n,g]`
  is just `LL @ Phi.T`.

The ELBO (data + assignment term, the entropies of `q(z)` and `q(φ)`, and the
stick-breaking KL) is evaluated each sweep; iteration stops when its relative change drops
below `tol`. The whole fit is repeated `n_restarts` times and the **best-ELBO** restart is
kept ([`fit_dpbb_vi`](model/dpbb_vi.py)).

### Initialisation (why restarts matter)

A flat start collapses every component onto the dominant clonal peak. Instead each restart
seeds the cluster CCFs by **k-means++** over the per-mutation MLE CCFs
`φ̂_n = grid[ argmax_g LL[n,g] ]`
([`_kmeanspp_centers`](model/dpbb_vi.py)), so components start at *distinct* prevalences. This
is the variational analogue of the multi-restart fix used by samplers.

## 5. The neutral 1/f tail component

Under neutral tumour evolution the subclonal mutation spectrum has a power-law tail
`M(f) ∝ f^(−γ)` (Williams et al., Nat Genet 2016). A plain Beta mixture "explains"
this smooth continuum by **over-splitting it into ghost clusters** — the dominant failure mode
on the simulated `T*` samples (1B collapses).

The fix ([`neutral_tail_grid`](model/dpbb_vi.py), enabled by `use_tail`) adds a **fixed**
power-law CCF distribution `g_tail[g] ∝ g^(−exponent)` (for `g ≥ tail_min_ccf`) as a separate
component (column 0). Crucially, its per-mutation contribution
is its true **marginal** log-likelihood

```
log Σ_g g_tail[g] · p(b_n | g)      (logsumexp)
```

**not** the mean-field `E[log p]` — because a diffuse distribution would always lose
the mean-field comparison to a peaked Beta. The tail absorbs the diffuse neutral mutations so
the Beta clusters stop over-splitting. It is **not** a subclonal lineage: in post-processing it
is dropped and its mutations are reassigned to the nearest Beta cluster (so it does not inflate
1B/1C but still contributes to 2A/2B).

## 6. Purity inference (sub-challenge 1A)

Purity is seeded from clonal diploid heterozygous SNVs, where `VAF ≈ ρ/2`
([`estimate_purity`](model/dpbb_vi.py)), or from the provided Battenberg `cellularity_ploidy.txt`.
With `purity_search=True`, [`01_cluster.py`](steps/01_cluster.py) then **infers** purity by
fitting the VI on a grid of `ρ` around the seed (`purity_span`, `purity_steps`) and keeping
the `ρ` that **maximises the ELBO** — a lower bound on `log p(data | ρ)`. The
selected `ρ` is the 1A prediction.

## 7. From clusters to SMC-Het estimate files (step 2)

[`02_scoring_files.py`](steps/02_scoring_files.py) maps the clustering onto the exact mutation
set and ordering of the scoring VCF, reproducing the DECODE `03_scoring_files.R` conventions so
comparisons are apples-to-apples:

- **VCF alignment** (`build_alignment`): each VCF mutation is matched by `chrom_pos`. Mutations
  present in our table take their cluster id / soft vector; mutations **absent** from our table
  default to **cluster 1** (clonal), with soft mass `[1, 0, …]`.
- **1A** — the purity.
- **1C** — tabulate per-cluster counts over the VCF-aligned assignment, keep clusters with
  ≥1 mutation; the SMC-Het *cellular prevalence* is `CCF × purity` (clonal CCF=1 → prevalence=
  purity).
- **1B** — the number of kept clusters.
- **2A** — the per-mutation cluster id, in VCF order.
- **2B** *(optional, `--with-2b`)* — the soft co-clustering matrix `CCM = P · Pᵀ`
  (probability that two mutations share a cluster), gzipped; skipped above `--max-2b-n` to avoid
  `O(N²)` blow-up.

## 8. Scoring (step 3)

[`03_score.py`](steps/03_score.py) runs the vendored Python 3 scorer
([`smc_het_eval/SMCScoring.py`](smc_het_eval/SMCScoring.py)) once per (sample, sub-challenge),
comparing each estimate file against the matching `*.truth.*` file (using the truth scoring
VCF where required), then aggregates everything into `scores_summary.csv`.

## References

- **PyClone-VI** — Gillis & Roth, *BMC Bioinformatics* 2020 (variational, CCF-grid trick).
- **Stick-breaking DP** — Sethuraman 1994; **mean-field DP** — Blei & Jordan 2006.
- **ICL** — Biernacki, Celeux & Govaert 2000; **ICL-guided merging** — Baudry et al. 2010.
- **Neutral tumour evolution (1/f)** — Williams et al., *Nat Genet* 2016; **MOBSTER** —
  Caravagna et al., *Nat Genet* 2020.
- **Power-law MLE** — Clauset, Shalizi & Newman 2009.
- **Coarsened posterior / penalty tempering** — Miller & Dunson 2019.
- **SMC-Het / DREAM** — Salcedo et al., *Nat Biotechnol* 2020 (the scoring harness in
  `smc_het_eval/`).
