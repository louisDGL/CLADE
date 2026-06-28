"""DECODE_2_2 paths & params (all derived from PROJECT_ROOT)."""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "dream_csv": os.path.join(PROJECT_ROOT, "DREAM_data"),         # input mutation tables
    "results":   os.path.join(PROJECT_ROOT, "results"),            # per-sample VI results (.npz)
    "truth":     os.path.join(PROJECT_ROOT, "scoring/truth"),      # ground truth (vendored in)
    "pred":      os.path.join(PROJECT_ROOT, "scoring/pred_soft"),  # estimate.{1A,1B,1C,2A,2B}
    "scores":    os.path.join(PROJECT_ROOT, "scoring/scores"),
}

PARAMS = {
    # --- model ---
    "n_grid":      100,     # CCF grid resolution
    "K_max":       12,      # DP truncation (max clusters)
    "alpha":       1.0,     # DP concentration
    "precision":   200.0,   # Beta-Binomial precision (overdispersion). None -> Binomial (over-splits).
    "n_restarts":  10,      # VI restarts (k-means++ seeded), keep best ELBO
    # --- neutral tail (MOBSTER 1/f component) — absorbs diffuse neutral mutations ---
    "use_tail":      True,  # add a fixed power-law CCF component (fixes 1B over-split of the tail)
    "tail_exponent": 2.0,   # 1/f^exponent (Williams neutral evolution -> 2.0)
    "tail_min_ccf":  0.03,  # power-law support lower bound (detection floor)
    "max_iter":    400,
    "tol":         1e-5,
    # --- purity ---
    "purity_mode":   "battenberg",  # seed: "battenberg" (cellularity_ploidy.txt) | "estimate" (from data)
    "purity_search": True,          # refine purity by maximising the VI ELBO over a grid (infers 1A)
    "purity_span":   0.12,          # +/- range around the seed for the ELBO search
    "purity_steps":  7,             # grid resolution for the search
    # --- cluster-number selection ---
    "selection":     "icl",      # "icl" (principled, ICL-guided merging, threshold-free) | "merge"
    "icl_ent_w":     0.25,       # entropy-term weight in the ICL merge; <1 rescues sparse subclones
                                 # (the neutral tail already guards over-split). 0.25 -> best 1B.
    "mass_floor_frac": 0.005,    # drop DP components below this fraction of mutations
    "merge_ccf":       0.12,     # only used when selection="merge" (the hand-tuned band-aid baseline)
}
