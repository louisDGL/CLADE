#!/usr/bin/env python3
"""
Step 4 (optional) — per-sample SFS-style fit plots, styled after DECODE's
`R/DECODE2_plots.r` (DECODE2_plot_SFS).

The reference panel shows the observed mutation spectrum as bars, with the *fitted model*
overlaid as stacked, per-component coloured areas (a colour line on each stacked top), a
horizontal legend on top, and the sample id as title. We reproduce that look from
DECODE_2_2's own output:

  * data        : histogram of the observed Variant Allele Frequency (thin black bars);
  * model       : the engine's posterior-predictive SFS, decomposed per cluster and drawn
                  as stacked coloured areas + colour lines. For cluster k (point CCF phi_k,
                  weight pi_k = counts_k / N), the expected count in a frequency bin is
                  pi_k * sum_n P(freq_n in bin | depth d_n, xi_n(phi_k), precision s),
                  with xi_n(phi) = rho*m_n*phi / (rho*CN_n + 2(1-rho)) the engine's
                  expected VAF and a Beta-Binomial(precision s) emission — exactly the
                  model fitted in model/dpbb_vi.py.

DECODE_2_2 has no SFS/ABC machinery (no inference-A/B/validation splits, no candidate-fit
grid, no selection-criterion boxplot, no read-count heatmap), so only this SFS panel is
reproduced. The neutral 1/f tail is a fitting device that postprocess reabsorbs into the
Beta clusters, so there is no separate grey tail band (this matches the reference's no-tail
"NT+N" panels; the engine's final model is clusters-only).

  python3 steps/04_plots.py                 # all samples -> plots/<sid>_sfs.png
  python3 steps/04_plots.py --samples P1-noXY S2-noXY
  python3 steps/04_plots.py --space ccf      # x-axis = CCF instead of VAF
  SLURM_ARRAY_TASK_ID=1 python3 steps/04_plots.py
"""
import os, sys, csv, argparse, math
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, PARAMS

# Reference palette (DECODE2_plots.r): per-cluster Okabe-Ito colours; data drawn in black.
CLUSTER_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
                  "#F0E442", "#000000", "#999933", "#882255", "#117733", "#332288"]
DATA_COLOR = "#1A1A1A"
N_BINS = 80


def load_table(path):
    """Read counts from a DREAM_data mutation table (mirrors steps/01_cluster.py)."""
    major, minor, ref, alt, mult = [], [], [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            major.append(float(row["major_cn"])); minor.append(float(row["minor_cn"]))
            ref.append(float(row["Ref_count"])); alt.append(float(row["Alt_count"]))
            mult.append(float(row.get("multiplicity", 1) or 1))
    major, minor = np.array(major), np.array(minor)
    alt, ref = np.array(alt), np.array(ref)
    return alt, alt + ref, major + minor, np.maximum(np.array(mult), 1.0)


def color_for(k):
    return CLUSTER_COLORS[k % len(CLUSTER_COLORS)]


def predicted_sfs(d, cn, mult, purity, ccf, counts, edges, precision, space):
    """Posterior-predictive expected counts per (cluster, bin) on the chosen frequency axis.

    Returns (K, n_bins). Each mutation's frequency under cluster k is approximated by a
    Gaussian with the Beta-Binomial mean/variance (exact moments of the emission fitted in
    model/dpbb_vi.py): for VAF, mean xi_n(phi_k) and var xi(1-xi)(d+s)/(d(s+1)); the bin
    masses are summed over mutations and weighted by pi_k = counts_k / N. (Gaussian is the
    smooth, fast stand-in for the per-read-count Beta-Binomial; visually identical.)
    """
    N = len(d); K = len(ccf)
    pred = np.zeros((K, len(edges) - 1))
    s = precision if precision is not None else 1e9          # ~Binomial when precision is None
    dd = np.maximum(d.astype(float), 1.0)
    factor = (purity * cn + 2.0 * (1.0 - purity)) / np.maximum(purity * mult, 1e-9)  # CCF = VAF*factor
    for k in range(K):
        xi = np.clip(ccf[k] / factor, 1e-6, 1 - 1e-6)        # expected VAF of cluster k per mut
        sd_vaf = np.sqrt(np.maximum(xi * (1 - xi) * (dd + s) / (dd * (s + 1)), 1e-12))
        mean = xi if space == "vaf" else np.full(N, ccf[k])
        sd = sd_vaf if space == "vaf" else sd_vaf * factor
        cdf = norm.cdf((edges[None, :] - mean[:, None]) / sd[:, None])  # (N, n_bins+1)
        pred[k] = (counts[k] / N) * np.diff(cdf, axis=1).sum(axis=0)
    return pred


def plot_sample(sid, space="vaf"):
    npz = os.path.join(PATHS["results"], f"{sid}_dpbbvi.npz")
    csvp = os.path.join(PATHS["dream_csv"], f"{sid}_mutation_table_with_multiplicity.csv")
    if not (os.path.exists(npz) and os.path.exists(csvp)):
        print(f"skip {sid}: missing npz or csv", flush=True)
        return
    z = np.load(npz, allow_pickle=True)
    ccf = np.atleast_1d(z["ccf"]); counts = np.atleast_1d(z["counts"]).astype(float)
    purity = float(z["purity"]); K = int(z["n_clusters"])
    b, dep, cn, mult = load_table(csvp)
    vaf = b / np.maximum(dep, 1)

    if space == "vaf":
        x = vaf
        x_hi = min(1.0, float(np.quantile(vaf, 0.999)) + 0.05)
        edges = np.linspace(0.0, x_hi, N_BINS + 1)
        xlabel = "Variant Allele Frequency"
        cluster_marks = None
    else:
        factor = (purity * cn + 2.0 * (1.0 - purity)) / np.maximum(purity * mult, 1e-9)
        x = np.clip(vaf * factor, 0.0, None)
        edges = np.linspace(0.0, 1.3, N_BINS + 1)
        xlabel = "Cancer Cell Fraction (CCF)"
        cluster_marks = ccf

    centers = 0.5 * (edges[:-1] + edges[1:]); binw = edges[1] - edges[0]
    data_counts, _ = np.histogram(x, bins=edges)
    pred = predicted_sfs(dep, cn, mult, purity, ccf, counts, edges,
                         PARAMS.get("precision"), space)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    # model: stacked coloured areas (alpha 0.5) + a colour line on each cumulative top edge
    cum = np.zeros_like(centers)
    for k in range(K):
        top = cum + pred[k]
        ax.fill_between(centers, cum, top, color=color_for(k), alpha=0.5, linewidth=0, zorder=2)
        ax.plot(centers, top, color=color_for(k), linewidth=2.2, zorder=3)
        cum = top
    if cluster_marks is not None:
        for k in range(K):
            ax.axvline(cluster_marks[k], color=color_for(k), linestyle="--",
                       linewidth=1.4, alpha=0.85, zorder=4)
    # data: thin black bars on top (half-bin width, like the reference's geom_bar)
    ax.bar(centers, data_counts, width=0.55 * binw, color=DATA_COLOR, linewidth=0, zorder=5)

    ax.set_xlim(edges[0], edges[-1]); ax.margins(y=0.02)
    ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel("Mutation count", fontsize=13)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=11)
    # purity / cluster-count info box, inside the axes (never collides with the header)
    ax.text(0.985, 0.96, f"purity (1A) = {purity:.3f}\n{K} cluster(s)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#444444",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dddddd", alpha=0.85))

    handles = [Line2D([0], [0], color=DATA_COLOR, lw=5, label="Data")]
    handles += [Patch(facecolor=color_for(k), edgecolor="none", alpha=0.65,
                      label=f"Cluster {k + 1}  (CCF {ccf[k]:.2f}, n={int(counts[k])})")
                for k in range(K)]
    ncol = min(K + 1, 4)
    nrow = math.ceil((K + 1) / ncol)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=ncol, fontsize=10, frameon=False, columnspacing=1.4, handlelength=1.4)
    # title above the legend: reserve vertical room that scales with the legend's row count
    ax.set_title(sid, fontsize=16, fontweight="bold", pad=12 + nrow * 20)
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{sid}_sfs.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"### PLOT {sid} -> {out} | K={K} purity={purity:.3f} "
          f"ccf=[{', '.join(f'{c:.3f}' for c in ccf)}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None, help="sample ids (default: all)")
    ap.add_argument("--space", choices=("vaf", "ccf"), default="vaf",
                    help="frequency axis (default: vaf, like the reference)")
    args = ap.parse_args()
    if args.samples:
        sids = args.samples
    else:
        npzs = sorted(f for f in os.listdir(PATHS["results"]) if f.endswith("_dpbbvi.npz"))
        sids = [f[:-len("_dpbbvi.npz")] for f in npzs]
        idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if idx is not None:
            sids = [sids[int(idx) - 1]]
    for sid in sids:
        plot_sample(sid, space=args.space)


if __name__ == "__main__":
    main()
