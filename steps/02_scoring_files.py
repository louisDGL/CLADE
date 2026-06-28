#!/usr/bin/env python3
"""
Step 2 — SMC-Het estimate files from the VI results, aligned to the scoring VCF.

Replicates DECODE2_joint/vignettes/03_scoring_files.R conventions EXACTLY so the
head-to-head is fair: VCF mutations absent from our table go to cluster 1; 1C counts
are tabulated over the VCF-aligned assignment; 2B (soft) = P @ P.T.

Output: scoring/pred_soft/<sid>/<sid>_estimate.{1A,1B,1C,2A[,2B.gz]}
"""
import os, sys, gzip, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS


def parse_scoring_vcf(path):
    keys = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            keys.append("%s_%s" % (f[0], f[1]))
    return keys


def build_alignment(npz, vcf_keys):
    chrom = npz["chrom"]; pos = npz["pos"]; hard = npz["hard"]; soft = npz["soft"]
    K = soft.shape[1]
    lut_clust = {}
    lut_soft = {}
    for i in range(len(chrom)):
        key = "%s_%s" % (chrom[i], pos[i])
        lut_clust[key] = int(hard[i]) + 1            # 1-based cluster id
        lut_soft[key] = soft[i]
    n = len(vcf_keys)
    assign = np.ones(n, dtype=int)                   # default cluster 1
    P = np.zeros((n, K)); P[:, 0] = 1.0              # default prob mass on clonal
    for j, key in enumerate(vcf_keys):
        if key in lut_clust:
            assign[j] = lut_clust[key]
            P[j] = lut_soft[key]
    return assign, P


def write_sample(sid, with_2b=False, max_2b_n=20000):
    res = os.path.join(PATHS["results"], f"{sid}_dpbbvi.npz")
    vcf = os.path.join(PATHS["truth"], sid, f"{sid}.truth.scoring_vcf.vcf")
    if not (os.path.exists(res) and os.path.exists(vcf)):
        print(f"  {sid}: SKIP (result or vcf missing)"); return
    npz = np.load(res, allow_pickle=True)
    purity = float(npz["purity"]); ccf = npz["ccf"]
    vcf_keys = parse_scoring_vcf(vcf)
    assign, P = build_alignment(npz, vcf_keys)

    out_dir = os.path.join(PATHS["pred"], sid); os.makedirs(out_dir, exist_ok=True)
    # 1A purity
    with open(os.path.join(out_dir, f"{sid}_estimate.1A.txt"), "w") as f:
        f.write("%.6f\n" % purity)
    # 1C architecture (tabulate over VCF assignment; keep clusters with >=1).
    # SMC-Het "cellular prevalence" = CCF x purity (clonal phi == purity), matching the
    # DECODE convention (clonal CCF == purity). Our model CCF is in [0,1] (clonal==1), so
    # multiply by purity here.
    counts = np.bincount(assign - 1, minlength=len(ccf))
    keep = np.where(counts >= 1)[0]
    with open(os.path.join(out_dir, f"{sid}_estimate.1C.txt"), "w") as f:
        for new_id, k in enumerate(keep, start=1):
            prevalence = float(np.clip(ccf[k] * purity, 0, 1))
            f.write("%d\t%d\t%.6f\n" % (new_id, counts[k], prevalence))
    # 1B count
    with open(os.path.join(out_dir, f"{sid}_estimate.1B.txt"), "w") as f:
        f.write("%d\n" % len(keep))
    # 2A per-mutation assignment (VCF order)
    with open(os.path.join(out_dir, f"{sid}_estimate.2A.txt"), "w") as f:
        f.write("\n".join(str(a) for a in assign) + "\n")
    # 2B soft co-clustering matrix (optional, disk-heavy)
    n = len(vcf_keys)
    if with_2b:
        if n > max_2b_n:
            print(f"  {sid}: 2B SKIPPED (N={n} > {max_2b_n})")
        else:
            ccm = P @ P.T
            out = os.path.join(out_dir, f"{sid}_estimate.2B.txt.gz")
            with gzip.open(out, "wt") as f:
                np.savetxt(f, ccm, fmt="%.6g", delimiter="\t")
    print(f"  {sid}: purity={purity:.3f} K={len(keep)} (N_vcf={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-2b", action="store_true", help="also build the 2B co-clustering matrix")
    ap.add_argument("--max-2b-n", type=int, default=20000)
    ap.add_argument("--samples", nargs="+", default=None)
    args = ap.parse_args()
    sids = args.samples or sorted(
        f[:-len("_dpbbvi.npz")] for f in os.listdir(PATHS["results"]) if f.endswith("_dpbbvi.npz"))
    print(f"Generating estimate files for {len(sids)} sample(s) | 2B={args.with_2b}")
    for sid in sids:
        write_sample(sid, with_2b=args.with_2b, max_2b_n=args.max_2b_n)


if __name__ == "__main__":
    main()
