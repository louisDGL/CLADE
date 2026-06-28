"""
Dirichlet-Process Beta-Binomial mixture, mean-field variational inference.

No external deps beyond numpy/scipy. The cluster CCF of each component is represented
as a categorical distribution over a fixed grid (PyClone-VI's tractability trick), which
makes the non-conjugate Beta-Binomial emission amenable to closed-form coordinate-ascent.

Generative model (per sample, purity rho given):
  v_k       ~ Beta(1, alpha)                 # stick-breaking, k=1..K_max (truncated DP)
  pi_k      = v_k * prod_{j<k}(1 - v_j)
  phi_k     ~ Categorical over CCF grid       # cluster cellular prevalence in (0,1]
  z_n       ~ Categorical(pi)
  xi_n(phi) = rho*mult_n*phi / (rho*CN_n + (1-rho)*2)     # expected VAF
  b_n       ~ Binomial(d_n, xi_n(phi_{z_n}))  # or Beta-Binomial(precision s)

Returns cluster CCFs, weights and per-mutation soft responsibilities.

Reading order (the pipeline calls only the last two):
  fit_dpbb_vi()  -> runs n_restarts of _fit_once() (the CAVI loop), keeps the best ELBO.
  postprocess()  -> drops the tail, prunes, selects the cluster number (_icl_merge), relabels.
Everything else (_expected_vaf, _emission_loglik, _stick_expectations, _icl) is a helper.

Array-shape conventions used throughout:
  N = #mutations,  G = #grid points (CCF resolution),  K = #mixture components.
  LL  : (N, G)  emission log-likelihood, mutation x grid-CCF       (shared by all clusters)
  Phi : (K, G)  each row = one cluster's categorical posterior over the CCF grid
  r   : (N, K)  soft responsibilities, each row sums to 1          (the assignment q(z_n))
"""
import numpy as np
from scipy.special import digamma, gammaln, logsumexp

_EPS = 1e-12


def estimate_purity(b, d, cn_total, mult):
    """Purity estimate from clonal diploid heterozygous SNVs (VAF ~ rho/2).

    A clonal mutation on one of the two copies of a diploid (CN=2, m=1) locus is seen, in
    tumour cells only, at VAF = rho/2 -> rho ~= 2*VAF. We take the median of the UPPER half
    of the VAFs (`v_high`) to lock onto that clonal peak and ignore the lower, subclonal/noise
    VAFs. If too few diploid sites exist (<20) we fall back to all mutations.
    """
    vaf = b / np.maximum(d, 1)
    diploid = (cn_total == 2) & (mult == 1)
    v = vaf[diploid] if diploid.sum() >= 20 else vaf
    if v.size == 0:
        return 0.5
    v_high = v[v > np.median(v)]            # keep the clonal (high-VAF) half
    if v_high.size == 0:
        v_high = v
    return float(np.clip(2.0 * np.median(v_high), 0.05, 1.0))


def _expected_vaf(phi_grid, cn_total, mult, purity):
    """xi[n, g] = expected VAF for mutation n if its cluster has CCF phi_grid[g].

    Builds the full (N, G) matrix by broadcasting: mult/cn vary over rows (mutations),
    phi over columns (grid). This is eq. xi_n(phi) = rho*m*phi / (rho*CN + 2(1-rho)); the
    denominator is the average local ploidy. Clipped off {0,1} so the logs downstream stay finite.
    """
    num = purity * mult[:, None] * phi_grid[None, :]          # (N, 1) * (1, G) -> (N, G)
    den = purity * cn_total[:, None] + (1.0 - purity) * 2.0   # (N, 1), per-mutation ploidy
    xi = num / np.maximum(den, _EPS)
    return np.clip(xi, 1e-6, 1.0 - 1e-6)


def _emission_loglik(b, d, xi, precision=None):
    """log P(b | d, xi)  ->  (N, G).  Binomial if precision is None, else Beta-Binomial.

    Computed once for every (mutation, grid-CCF) pair and then SHARED by all clusters -- a
    cluster only differs through its grid posterior Phi, so this matrix is the workhorse.
    Constants that don't depend on xi (the binomial coefficient C(d,b)) are dropped: they
    cancel in every softmax / ELBO comparison.

    Principale source de rapidité du moteur
    """
    b = b[:, None]; d = d[:, None]                            # -> (N, 1), broadcast against xi (N, G)
    if precision is None:                       # Binomial (drop the constant binom coeff)
        return b * np.log(xi) + (d - b) * np.log1p(-xi)
    # Beta-Binomial with shape (a, bb) = (xi*s, (1-xi)*s): the log-pmf is
    #   logB(b+a, d-b+bb) - logB(a, bb)   with  logB(x,y)=gammaln(x)+gammaln(y)-gammaln(x+y).
    # Expanded below (the +C(d,b) constant is again omitted).
    a = xi * precision; bb = (1.0 - xi) * precision
    return (gammaln(a + bb) - gammaln(a) - gammaln(bb)              # -logB(a, bb)
            + gammaln(b + a) + gammaln(d - b + bb) - gammaln(d + a + bb))  # +logB(b+a, d-b+bb)


def _stick_expectations(a_stick, b_stick):
    """Expected log stick-breaking weights from q(v_k)=Beta(a_k, b_k), via digamma identities.

    For v~Beta(a,b):  E[log v] = psi(a)-psi(a+b),  E[log(1-v)] = psi(b)-psi(a+b).
    Then E[log pi_k] = E[log v_k] + sum_{j<k} E[log(1-v_j)] -- the prefix sum of E[log(1-v)]
    EXCLUDING the current k, built here as [0, cumsum(...)[:-1]] (a shift-by-one cumulative sum).
    """
    dig_ab = digamma(a_stick + b_stick)
    Elog_v = digamma(a_stick) - dig_ab
    Elog_1mv = digamma(b_stick) - dig_ab
    Elog_pi = Elog_v + np.concatenate([[0.0], np.cumsum(Elog_1mv)[:-1]])   # add sum over j<k
    return Elog_pi, Elog_v, Elog_1mv


def _kmeanspp_centers(ccf_mle, K, rng):
    """k-means++ seeding of K cluster CCF centers over per-mutation MLE CCFs, so components
    start at DISTINCT prevalences instead of all collapsing onto the dominant clonal peak.

    Standard k-means++: pick the first center at random, then each subsequent center with
    probability proportional to its squared distance (`d2`) to the nearest existing center --
    biasing new seeds towards under-covered CCF regions (e.g. faint subclones).
    """
    N = len(ccf_mle)
    centers = [float(ccf_mle[rng.integers(N)])]
    while len(centers) < K:
        d2 = np.min(np.stack([(ccf_mle - c) ** 2 for c in centers]), axis=0)  # nearest-center dist^2
        tot = d2.sum()
        if tot <= 0:                                          # all points coincide with a center
            centers.append(float(ccf_mle[rng.integers(N)]))
        else:
            centers.append(float(ccf_mle[rng.choice(N, p=d2 / tot)]))  # sample prop. to d2
    return np.array(centers[:K])


def _fit_once(LL, log_prior_phi, K_max, alpha, max_iter, tol, rng, tail_grid=None):
    """One CAVI run: coordinate-ascent on the ELBO from one (random) k-means++ initialisation.

    Sweeps the three blocks {q(phi_k), q(v_k), q(z_n)} in closed form until the ELBO converges.
    With a tail, component 0 is the fixed neutral tail and components 1.. are the Beta clusters.
    """
    N, G = LL.shape
    grid = (np.arange(G) + 0.5) / G                          # grid CCF values phi_g = (g+0.5)/G
    # --- spread initialisation: seed cluster CCFs by k-means++ on per-mutation MLE CCFs,
    #     so components start at DISTINCT prevalences instead of all collapsing onto the
    #     dominant (clonal) peak. This is the variational analogue of the SMC restart fix.
    ccf_mle = grid[LL.argmax(axis=1)]                        # per-mutation argmax-CCF (the MLE)
    centers = _kmeanspp_centers(ccf_mle, K_max, rng)
    sigma = 0.03
    # each cluster's initial grid posterior = a narrow Gaussian bump centred on its seed CCF
    Phi = np.exp(-0.5 * ((grid[None, :] - centers[:, None]) / sigma) ** 2)  # (K_max, G) Beta clusters
    Phi /= np.maximum(Phi.sum(axis=1, keepdims=True), _EPS)  # normalise each row to a distribution
    # --- optional neutral-tail component (MOBSTER 1/f): a FIXED power-law CCF distribution
    #     kept as a SEPARATE column 0 whose per-mutation contribution is its true MARGINAL
    #     log-likelihood  log sum_g g_tail[g] p(b_n|g)  (NOT the mean-field E[log p], which
    #     would always lose to a peaked Beta). It absorbs the diffuse neutral mutations so the
    #     Beta clusters stop over-splitting the tail.
    has_tail = tail_grid is not None
    Elik_tail = (logsumexp(np.log(tail_grid + _EPS)[None, :] + LL, axis=1)   # marginal, not E[log p]
   
                 if has_tail else None)          # (N,)
    K = K_max + (1 if has_tail else 0)
    # seed the first responsibilities from the k-means++ init Phi (else the first phi-update,
    # computed from a uniform r, would average all clusters to one and collapse the fit)
    Elik0 = LL @ Phi.T                                       # (N, K_max): E_q(phi)[log p] per cluster
    if has_tail:
        Elik0 = np.column_stack([Elik_tail, Elik0])          # prepend the tail column (component 0)
    logr = np.log(1.0 / K) + Elik0                           # uniform pi prior + evidence
    r = np.exp(logr - logsumexp(logr, axis=1, keepdims=True))  # softmax over components -> (N, K)
    elbo_old = -np.inf
    for it in range(max_iter):
        # --- q(phi_k) for the Beta clusters (tail excluded). For each cluster, accumulate the
        #     responsibility-weighted emission over mutations, then softmax over the grid:
        #     logPhi[k,g] = log(1/G) + sum_n r[n,k] LL[n,g]   ==   (r_beta^T @ LL)[k,g] + const.
        r_beta = r[:, 1:] if has_tail else r
        logPhi = log_prior_phi + r_beta.T @ LL               # (K_max, G)
        logPhi -= logsumexp(logPhi, axis=1, keepdims=True)   # normalise each cluster over the grid
        Phi = np.exp(logPhi)
        # --- stick-breaking q(v_k) over all K components. N_k = expected mass of component k;
        #     q(v_k)=Beta(a_k,b_k) with a_k=1+N_k and b_k=alpha+sum_{j>k} N_j.
        Nk = r.sum(axis=0)                                   # (K,) component masses
        a_stick = 1.0 + Nk
        Ngt = np.concatenate([np.cumsum(Nk[::-1])[::-1][1:], [0.0]])  # sum_{j>k} Nj (reversed cumsum)
        b_stick = alpha + Ngt
        Elog_pi, Elog_v, Elog_1mv = _stick_expectations(a_stick, b_stick)
        # --- q(z_n): Beta cols use E[log p]=sum_g Phi LL (= LL @ Phi^T); tail col uses the
        #     marginal log-lik; then r[n,k] = softmax_k( E[log pi_k] + evidence[n,k] ).
        Elik_beta = LL @ Phi.T                  # (N, K_max)
        Elik = np.column_stack([Elik_tail, Elik_beta]) if has_tail else Elik_beta
        # log posterior = log prior + log vraisemblance
        logr = Elog_pi[None, :] + Elik
        logr -= logsumexp(logr, axis=1, keepdims=True)
        r = np.exp(logr)
        # --- ELBO = E_q[log p(b,z,v,phi)] - E_q[log q], decomposed into:
        elbo = np.sum(r * (Elog_pi[None, :] + Elik))                           # data + assignment
        elbo -= np.sum(r * np.where(r > _EPS, np.log(r + _EPS), 0.0))          # + H[q(z)]
        elbo += -np.sum(Phi * np.where(Phi > _EPS, np.log(Phi + _EPS), 0.0))   # + H[q(phi_beta)]
        elbo += Phi.shape[0] * log_prior_phi                                   # + uniform phi prior
        # --- KL( q(v_k)=Beta(a_k,b_k) || prior p(v_k)=Beta(1, alpha) ), one value per stick k
        kl_v = ((a_stick - 1.0) * Elog_v                                        # (a-a0)*E[log v],    a0=1
                + (b_stick - alpha) * Elog_1mv                                  # (b-b0)*E[log(1-v)], b0=alpha
                - (gammaln(a_stick) + gammaln(b_stick) - gammaln(a_stick + b_stick))  # - log B(a,b)
                + (gammaln(1.0) + gammaln(alpha) - gammaln(1.0 + alpha)))             # + log B(1, alpha)
        elbo -= np.sum(kl_v)                                                   # - sum_k KL stick-breaking
        # Convergence test
        if np.isfinite(elbo) and abs(elbo - elbo_old) < tol * (abs(elbo_old) + 1.0):
            break                                            # relative-ELBO convergence
        elbo_old = elbo
    ccf_beta = Phi @ grid                                    # point CCF per cluster = E_grid[phi]
    ccf = np.concatenate([[float(tail_grid @ grid)], ccf_beta]) if has_tail else ccf_beta
    return dict(r=r, ccf=ccf, elbo=float(elbo), has_tail=has_tail)


def neutral_tail_grid(phi_grid, exponent=2.0, min_ccf=0.03):
    """Fixed power-law (1/f^exponent) CCF distribution for the neutral component (MOBSTER).

    Neutral evolution predicts a 1/f^exponent subclonal tail (Williams 2016); this is its
    shape on the grid, zeroed below min_ccf (a detection floor) and normalised to a distribution.
    """
    g = np.where(phi_grid >= min_ccf, phi_grid ** (-exponent), 0.0)
    s = g.sum()
    return g / s if s > 0 else np.full_like(phi_grid, 1.0 / len(phi_grid))


def fit_dpbb_vi(b, d, cn_total, mult, purity, n_grid=100, K_max=10, alpha=1.0,
                precision=None, max_iter=400, tol=1e-5, n_restarts=10, seed=0,
                use_tail=False, tail_exponent=2.0, tail_min_ccf=0.03):
    """Fit the DP Beta-Binomial mixture (+ optional neutral 1/f tail); keep best-ELBO restart.

    Precomputes the (N, G) emission ONCE (the expensive part), then runs n_restarts CAVI fits
    from different k-means++ seeds and returns the highest-ELBO one (multimodal objective).
    """
    b = np.asarray(b, float); d = np.asarray(d, float)
    cn_total = np.asarray(cn_total, float); mult = np.asarray(mult, float)
    phi_grid = (np.arange(1, n_grid + 1) - 0.5) / n_grid     # CCF grid centres in (0,1]
    xi = _expected_vaf(phi_grid, cn_total, mult, purity)     # (N, G) expected VAFs
    LL = _emission_loglik(b, d, xi, precision)            # (N, G), shared by all clusters
    log_prior_phi = -np.log(n_grid)                         # uniform prior over the grid: log(1/G)
    tail_grid = neutral_tail_grid(phi_grid, tail_exponent, tail_min_ccf) if use_tail else None
    best = None
    for s in range(n_restarts):
        rng = np.random.default_rng(seed + s)               # deterministic per restart
        fit = _fit_once(LL, log_prior_phi, K_max, alpha, max_iter, tol, rng, tail_grid=tail_grid)
        if best is None or fit["elbo"] > best["elbo"]:
            best = fit                                       # keep the best-ELBO restart
    best["purity"] = purity
    best["LL"] = LL                # (N,G) emission log-lik, kept for in-memory ICL selection
    return best


def _gidx(c, G):
    """Map a continuous CCF value c in (0,1] to its nearest grid index (inverse of phi_g)."""
    return int(min(G - 1, max(0, round(c * G - 0.5))))


def _icl(R, clog, pi, N, pen_w=1.0, ent_w=1.0):
    """Integrated Completed Likelihood (higher = better): observed log-lik − penalty −
    assignment entropy. The entropy term penalises overlapping (over-split) clusters.

    N-aware penalty: the K−1 mixture weights are informed by the WHOLE sample (log N), but
    each cluster's CCF is informed only by the N_k mutations assigned to it (log N_k). This
    stops a small real subclone (N_k≈200) from being penalised by log(N≈138000) in big
    clonal-dominated samples — the cause of ICL under-clustering at large N.

    pen_w (<1) tempers the penalty (coarsened-posterior style, Miller-Dunson 2019) to offset
    model misspecification — BIC over-penalises when the emission/tail are imperfect.

    ent_w (<1) tempers the assignment-entropy term. The entropy double-counts over-split
    protection once a neutral 1/f tail already absorbs the diffuse continuum: at full weight it
    also merges away SPARSE real subclones (few low-depth mutations → ambiguous assignment →
    high entropy). With the tail active, ent_w<1 recovers those subclones (fixes the dominant
    under-clustering of 1B) without reintroducing continuum over-split (the tail handles that).

    Inputs: R (N,K) responsibilities, clog (N,K) emission log-lik at each cluster's POINT CCF,
    pi (K,) weights. (clog uses point CCFs so a merged cluster can be re-scored cheaply.)
    """
    logpi = np.log(pi + _EPS)
    ell = logsumexp(logpi[None, :] + clog, axis=1).sum()       # observed data log-likelihood
    ent = -np.sum(R * np.where(R > _EPS, np.log(R + _EPS), 0.0))  # assignment entropy
    K = R.shape[1]
    Nk = R.sum(axis=0)
    penalty = 0.5 * ((K - 1) * np.log(max(N, 2)) + np.sum(np.log(np.maximum(Nk, 2.0))))
    return ell - pen_w * penalty - ent_w * ent


def _icl_merge(R, ccf_k, LL, pen_w=1.0, ent_w=1.0):
    """Greedy ICL-guided agglomerative merging of mixture components (Baudry et al. 2010):
    repeatedly merge the pair that most improves ICL; stop when no merge improves it. A cluster
    contributes its likelihood at its POINT CCF (mass-weighted mean): merging two distinct CCFs
    yields one in between that fits neither well -> the likelihood drop stops the merge, while
    over-split (near-equal CCF) pairs merge freely. The principled replacement for a fixed merge.

    ent_w tempers the entropy term of the ICL (see _icl): <1 merges less aggressively, which
    rescues sparse real subclones when a neutral tail already guards against over-split.
    """
    N, G = LL.shape
    ccf_k = np.array(ccf_k, float)
    clog = LL[:, [_gidx(c, G) for c in ccf_k]]     # (N, K) likelihood at each cluster's point CCF
    pi = R.sum(axis=0) / N
    cur = _icl(R, clog, pi, N, pen_w, ent_w)       # current score
    while R.shape[1] > 1:
        K = R.shape[1]
        best_gain, best_state = 0.0, None          # only accept strictly-improving merges (>0)
        for j in range(K):                         # try every unordered pair (j, l)
            for l in range(j + 1, K):
                wj, wl = pi[j], pi[l]
                cm = (wj * ccf_k[j] + wl * ccf_k[l]) / max(wj + wl, _EPS)  # mass-weighted merged CCF
                cols = [c for c in range(K) if c not in (j, l)]           # untouched clusters
                R2 = np.column_stack([R[:, cols], R[:, j] + R[:, l]])      # sum the two columns
                ccf2 = np.concatenate([ccf_k[cols], [cm]])
                clog2 = np.column_stack([clog[:, cols], LL[:, _gidx(cm, G)]])  # re-score at merged CCF
                pi2 = R2.sum(axis=0) / N
                gain = _icl(R2, clog2, pi2, N, pen_w, ent_w) - cur
                if gain > best_gain:
                    best_gain, best_state = gain, (R2, ccf2, clog2, pi2)
        if best_state is None:                     # no merge helps -> stop
            break
        R, ccf_k, clog, pi = best_state            # commit the best merge and continue
        cur += best_gain
    return R, ccf_k


def postprocess(fit, mass_floor_frac=0.01, merge_ccf=0.0, selection="icl", icl_pen_w=1.0,
                icl_ent_w=1.0):
    """Prune empty DP components, select the cluster number (ICL or CCF-gap merge), relabel.

    selection="icl"   -> principled ICL-guided merging (Baudry 2010); ignores merge_ccf.
    selection="merge" -> fixed CCF-gap merge (the band-aid baseline).

    Returns hard assignment (1..K'), soft responsibilities over kept clusters,
    cluster CCFs and per-cluster mutation counts.

    If the fit has a neutral-tail component (index 0), it is dropped here: it is NOT a
    subclonal lineage (not counted in 1B/1C), and its diffuse mutations are reassigned to
    the nearest Beta cluster for 2A/2B (via responsibility renormalisation below). Pruning
    of Beta clusters uses their tail-excluded mass, so neutral over-splits vanish.
    """
    r = fit["r"]; ccf = fit["ccf"]; N = r.shape[0]
    # --- 1) drop the neutral tail (component 0); record its mass for diagnostics
    tail_frac = 0.0
    if fit.get("has_tail", False):
        tail_frac = float(r[:, 0].sum() / N)
        r = r[:, 1:]; ccf = ccf[1:]
    # --- 2) prune Beta components whose mass is below the floor (empty DP sticks / dust)
    Nk = r.sum(axis=0)
    keep = np.where(Nk >= max(1.0, mass_floor_frac * N))[0]
    if keep.size == 0:                                       # degenerate guard: keep the heaviest
        keep = np.array([int(np.argmax(Nk))])
    rk = r[:, keep]
    ck = ccf[keep]
    # --- 3) choose the number of clusters
    if selection == "icl" and rk.shape[1] > 1:
        # ICL-guided merging: principled cluster-number control (replaces the merge hack).
        rk, ck = _icl_merge(rk, ck, fit["LL"], pen_w=icl_pen_w, ent_w=icl_ent_w)
    elif merge_ccf > 0 and ck.size > 1:
        # baseline: greedily group clusters whose CCFs are within merge_ccf of each other
        order = np.argsort(-ck)
        ck = ck[order]; rk = rk[:, order]
        groups = [[0]]
        for i in range(1, ck.size):
            if ck[i - 1] - ck[i] < merge_ccf:               # close enough -> same group
                groups[-1].append(i)
            else:
                groups.append([i])
        merged_r = np.column_stack([rk[:, g].sum(axis=1) for g in groups])
        merged_ccf = np.array([np.average(ck[g], weights=(rk[:, g].sum(axis=0) + _EPS))
                               for g in groups])
        rk = merged_r; ck = merged_ccf
    # --- 4) renormalise, hard assign, drop clusters that end up with zero hard members
    rk = rk / np.maximum(rk.sum(axis=1, keepdims=True), _EPS)  # rows back to distributions
    hard = rk.argmax(axis=1)                                 # hard label = most-likely cluster
    counts = np.bincount(hard, minlength=ck.size)
    nonempty = np.where(counts > 0)[0]                       # clusters that won at least one mutation
    remap = {old: new for new, old in enumerate(nonempty)}   # compact the surviving labels
    hard = np.array([remap[h] for h in hard])
    rk = rk[:, nonempty]
    rk = rk / np.maximum(rk.sum(axis=1, keepdims=True), _EPS)
    ck = ck[nonempty]
    # --- 5) sort clusters by descending CCF (clonal first) for stable, comparable labels
    order = np.argsort(-ck)
    inv = np.argsort(order)                                  # inverse permutation to relabel `hard`
    hard = inv[hard]
    rk = rk[:, order]
    ck = ck[order]
    counts = np.bincount(hard, minlength=ck.size)
    return dict(hard=hard, soft=rk, ccf=np.clip(ck, 0.0, 1.0), counts=counts,
                n_clusters=int(ck.size), tail_frac=tail_frac)
