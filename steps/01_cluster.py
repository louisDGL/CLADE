#!/usr/bin/env python3
"""
Step 1 — DP Beta-Binomial VI clustering (one sample per SLURM array task, or all if no index).

Input  : DREAM_data/<sid>_mutation_table_with_multiplicity.csv
Output : results/<sid>_dpbbvi.npz  (chrom,pos,hard,soft,ccf,counts,purity,n_clusters,elbo)
"""
import os, sys, csv, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, PARAMS
from model.dpbb_vi import fit_dpbb_vi, postprocess, estimate_purity


def load_table(path):
    chrom, pos, major, minor, ref, alt, mult = [], [], [], [], [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            chrom.append(row["Chromosome"]); pos.append(int(row["Position"]))
            major.append(float(row["major_cn"])); minor.append(float(row["minor_cn"]))
            ref.append(float(row["Ref_count"])); alt.append(float(row["Alt_count"]))
            mult.append(float(row.get("multiplicity", 1) or 1))
    major = np.array(major); minor = np.array(minor)
    return (np.array(chrom), np.array(pos),
            np.array(alt), np.array(alt) + np.array(ref),
            major + minor, np.maximum(np.array(mult), 1.0))


def battenberg_purity(sid):
    p = os.path.join(PATHS["truth"], sid, f"{sid}.cellularity_ploidy.txt")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        next(fh); cell = float(next(fh).split()[0])
    return float(np.clip(cell, 0.05, 1.0))


def run_sample(sid):
    t0 = time.time()
    csv_path = os.path.join(PATHS["dream_csv"], f"{sid}_mutation_table_with_multiplicity.csv")
    chrom, pos, b, d, cn_total, mult = load_table(csv_path)

    if PARAMS["purity_mode"] == "battenberg":
        seed = battenberg_purity(sid) or estimate_purity(b, d, cn_total, mult)
    else:
        seed = estimate_purity(b, d, cn_total, mult)

    def _fit(rho):
        return fit_dpbb_vi(b, d, cn_total, mult, rho,
                           n_grid=PARAMS["n_grid"], K_max=PARAMS["K_max"], alpha=PARAMS["alpha"],
                           precision=PARAMS["precision"], max_iter=PARAMS["max_iter"],
                           tol=PARAMS["tol"], n_restarts=PARAMS["n_restarts"],
                           use_tail=PARAMS["use_tail"], tail_exponent=PARAMS["tail_exponent"],
                           tail_min_ccf=PARAMS["tail_min_ccf"])

    if PARAMS.get("purity_search", False):
        # infer purity: pick the grid value maximising the VI ELBO (a lower bound on
        # log p(data | purity)), seeded at battenberg/estimate.
        lo = max(0.05, seed - PARAMS["purity_span"]); hi = min(1.0, seed + PARAMS["purity_span"])
        grid = np.linspace(lo, hi, PARAMS["purity_steps"])
        fits = [(_fit(rho), rho) for rho in grid]
        fit, purity = max(fits, key=lambda fr: fr[0]["elbo"])
        fit["purity"] = purity
    else:
        purity = seed
        fit = _fit(purity)
    post = postprocess(fit, mass_floor_frac=PARAMS["mass_floor_frac"],
                       merge_ccf=PARAMS["merge_ccf"], selection=PARAMS["selection"],
                       icl_ent_w=PARAMS["icl_ent_w"])

    os.makedirs(PATHS["results"], exist_ok=True)
    out = os.path.join(PATHS["results"], f"{sid}_dpbbvi.npz")
    np.savez_compressed(out, chrom=chrom, pos=pos, hard=post["hard"], soft=post["soft"],
                        ccf=post["ccf"], counts=post["counts"], purity=purity,
                        n_clusters=post["n_clusters"], elbo=fit["elbo"])
    print(f"### DONE {sid} | purity={purity:.3f} | n_clusters={post['n_clusters']} "
          f"| tail_frac={post.get('tail_frac', 0):.2f} "
          f"| ccf=[{', '.join(f'{c:.3f}' for c in post['ccf'])}] | {time.time()-t0:.1f}s", flush=True)
    return post["n_clusters"]


def main():
    csvs = sorted(f for f in os.listdir(PATHS["dream_csv"])
                  if f.endswith("_mutation_table_with_multiplicity.csv"))
    sids = [f[:-len("_mutation_table_with_multiplicity.csv")] for f in csvs]
    idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx is not None:
        sid = sids[int(idx) - 1]
        print(f"### cluster | task {idx} | sample {sid} | host {os.uname().nodename} | {time.ctime()}", flush=True)
        run_sample(sid)
    else:
        for sid in sids:
            run_sample(sid)


if __name__ == "__main__":
    main()
