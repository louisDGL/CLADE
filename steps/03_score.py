#!/usr/bin/env python3
# =============================================================================
# Step 4 — SMC-Het scoring driver
#
# Scores the predictions of THIS project using the SMC-Het scoring harness,
# which is VENDORED into this repo at smc_het_eval/ (no external dependency).
#
# Loops over each sample x sub-challenge and calls
# smc_het_eval/SMCScoring.py (Python 3) to compare prediction
# vs truth; then aggregates all scores into scores_summary.csv.
#
# ▶ Run (from the project root):
#     python3 vignettes/04_score.py soft            # typical case
#     python3 vignettes/04_score.py [soft|hard]
#             [--challenges 1A 1B 1C 2A 2B]   (default)
#             [--scores-dir DIR]           (default: scoring/scores)
#             [--samples P10-noXY ...]     (default: all)
# =============================================================================
import argparse
import datetime
import os
import subprocess
import sys


class _Tee:
    """Duplicates writes to two streams (e.g. stdout + log file)."""
    def __init__(self, primary, secondary):
        self._p, self._s = primary, secondary
    def write(self, data):
        self._p.write(data); self._s.write(data)
    def flush(self):
        self._p.flush(); self._s.flush()
    def fileno(self):
        return self._p.fileno()

# --- Paths: derived from this script's location (self-contained) -------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The SMC-Het scoring harness is VENDORED into this repo at smc_het_eval/.
# SMCScoring.py imports `smc_het_eval...` (absolute package), so it runs with
# PROJECT_ROOT (which contains smc_het_eval/) on PYTHONPATH.
SMC_HARNESS_ROOT = PROJECT_ROOT
# The vendored scorer now runs under Python 3 (this same interpreter by default).
# Override with $SMCHET_PY only if you must point at a specific python.
PYTHON       = os.environ.get("SMCHET_PY", sys.executable)
SMCSCORING   = os.path.join(SMC_HARNESS_ROOT, "smc_het_eval", "SMCScoring.py")

PATHS = {
    "truth":        os.path.join(PROJECT_ROOT, "scoring/truth"),
    "scoring_hard": os.path.join(PROJECT_ROOT, "scoring/pred_hard"),
    "scoring_soft": os.path.join(PROJECT_ROOT, "scoring/pred_soft"),
    "scores":       os.path.join(PROJECT_ROOT, "scoring/scores"),
}

# Per sub-challenge: file extension, and whether a --vcf is required.
# (consistent with challengeMapping in SMCScoring.py)
CHALLENGES = {
    "1A": {"ext": "txt", "vcf": False},
    "1B": {"ext": "txt", "vcf": False},
    "1C": {"ext": "txt", "vcf": True},
    "2A": {"ext": "txt", "vcf": True},
    "2B": {"ext": "gz",  "vcf": True},
}
DEFAULT_CHALLENGES = ["1A", "1B", "1C", "2A", "2B"]


def score_one(sid, ch, scoring_dir, truth_dir, scores_dir, max_2b_n=None):
    """Score one (sample, challenge). Returns the score string, or None if skipped."""
    spec     = CHALLENGES[ch]
    ext      = spec["ext"]
    pred     = os.path.join(scoring_dir, sid, "%s_estimate.%s.%s" % (sid, ch, ext))
    truth    = os.path.join(truth_dir,   sid, "%s.truth.%s.%s"    % (sid, ch, ext))
    vcf      = os.path.join(truth_dir,   sid, "%s.truth.scoring_vcf.vcf" % sid)
    out_dir  = os.path.join(scores_dir, sid)
    out_txt  = os.path.join(out_dir, "%s.%s.txt" % (sid, ch))
    out_log  = out_txt + ".log"

    if not os.path.exists(pred) or not os.path.exists(truth):
        print("  [%s] %s : SKIPPED (pred or truth missing)" % (sid, ch))
        return None
    if spec["vcf"] and not os.path.exists(vcf):
        print("  [%s] %s : SKIPPED (vcf missing)" % (sid, ch))
        return None

    # 2B: SMCScoring loads the whole N×N matrix into RAM. Skip the giants
    # to prevent the OOM killer from killing the entire process.
    if ch == "2B" and max_2b_n is not None:
        with open(vcf) as _f:
            n = sum(1 for line in _f if not line.startswith("#"))
        if n > max_2b_n:
            print("  [%s] %s : SKIPPED (N=%d > threshold %d)" % (sid, ch, n, max_2b_n))
            return None

    os.makedirs(out_dir, exist_ok=True)
    cmd = [PYTHON, SMCSCORING, "-c", ch,
           "--predfiles", pred, "--truthfiles", truth]
    if spec["vcf"]:
        cmd += ["--vcf", vcf]
    cmd += ["-o", out_txt]

    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    with open(out_log, "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=PROJECT_ROOT, env=env)
    if rc != 0 or not os.path.exists(out_txt):
        print("  [%s] %s : FAILED (code %d, see %s)" % (sid, ch, rc, out_log))
        return None

    with open(out_txt) as fh:
        value = fh.read().strip()
    print("  [%s] %s : %s" % (sid, ch, value))
    return value


def main():
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    _lf = open(os.path.join(log_dir, "04_score.log"), "w", buffering=1)
    _lf.write("### %s\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sys.stdout = _Tee(sys.__stdout__, _lf)
    sys.stderr = _Tee(sys.__stderr__, _lf)

    ap = argparse.ArgumentParser(description="SMC-Het scoring driver (step 4).")
    ap.add_argument("mode", nargs="?", default="soft", choices=["soft", "hard"],
                    help="Which step 3 output to score (default: soft).")
    ap.add_argument("--challenges", nargs="+", default=DEFAULT_CHALLENGES,
                    choices=list(CHALLENGES.keys()))
    ap.add_argument("--scores-dir", default=PATHS["scores"],
                    help="Output directory (default: scoring/scores).")
    ap.add_argument("--samples", nargs="+", default=None,
                    help="Restrict to these samples (default: all).")
    ap.add_argument("--max-2b-n", type=int, default=20000,
                    help="Skip 2B scoring if N > threshold (avoids OOM). Default: 20000. 0=no limit.")
    args = ap.parse_args()

    scoring_dir = PATHS["scoring_hard"] if args.mode == "hard" else PATHS["scoring_soft"]
    truth_dir   = PATHS["truth"]
    scores_dir  = args.scores_dir

    if not os.path.isdir(scoring_dir):
        sys.exit("Scoring directory not found: %s (run step 3 first)" % scoring_dir)
    if not os.path.exists(SMCSCORING):
        sys.exit("Scoring harness not found: %s" % SMCSCORING)

    samples = args.samples or sorted(
        d for d in os.listdir(scoring_dir)
        if os.path.isdir(os.path.join(scoring_dir, d))
    )

    print("Mode=%s | challenges=%s | %d sample(s) | output=%s\n"
          % (args.mode, ",".join(args.challenges), len(samples), scores_dir))

    results = {}  # sid -> {ch: value}
    for sid in samples:
        print("== %s ==" % sid)
        results[sid] = {}
        max_2b_n = None if args.max_2b_n == 0 else args.max_2b_n
        for ch in args.challenges:
            value = score_one(sid, ch, scoring_dir, truth_dir, scores_dir, max_2b_n=max_2b_n)
            if value is not None:
                results[sid][ch] = value

    write_summary(scores_dir)


def write_summary(scores_dir):
    """(Re)generates scores_summary.csv by scanning the disk: additive, does not
    lose any already-scored challenge. One column per present challenge."""
    canonical = ["1A", "1B", "1C", "2A", "2B"]
    os.makedirs(scores_dir, exist_ok=True)
    sids = sorted(d for d in os.listdir(scores_dir)
                  if os.path.isdir(os.path.join(scores_dir, d)))

    def score_path(sid, ch):
        return os.path.join(scores_dir, sid, "%s.%s.txt" % (sid, ch))

    present = [ch for ch in canonical
               if any(os.path.exists(score_path(s, ch)) for s in sids)]

    summary = os.path.join(scores_dir, "scores_summary.csv")
    with open(summary, "w") as fh:
        fh.write("sample," + ",".join(present) + "\n")
        for sid in sids:
            cells = []
            for ch in present:
                p = score_path(sid, ch)
                v = ""
                if os.path.exists(p):
                    with open(p) as f:
                        v = f.read().strip()
                cells.append('"%s"' % v)
            fh.write(sid + "," + ",".join(cells) + "\n")
    print("\n-> %s" % summary)


if __name__ == "__main__":
    main()
