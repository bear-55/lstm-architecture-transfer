"""Frozen selection-transfer estimand, with whole-seed and HUC2 resampling."""
import numpy as np


def validate(p, q, regions):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[0] < 3:
        raise ValueError('Require matching seed-by-basin matrices and at least three seeds')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or len(regions) != p.shape[1]:
        raise ValueError('No silent missing/nonfinite or region filtering')
    return p, q, np.unique(np.asarray(regions), return_inverse=True)[1]


def weighted(p, q, seed_weights, basin_weights):
    choose = (p > 0).astype(float)
    sw, bw = np.asarray(seed_weights, float), np.asarray(basin_weights, float)
    if sw.sum() <= 0 or bw.sum() <= 0:
        raise ValueError('Empty resample')
    bw = bw / bw.sum()
    matrix = np.einsum('ib,jb,b->ij', choose, q, bw)
    diag = np.diag(matrix)
    same = sw @ diag / sw.sum()
    pair_weights = np.outer(sw, sw)
    np.fill_diagonal(pair_weights, 0)  # distinct ORIGINAL seed identities only
    if pair_weights.sum() <= 0:
        raise ValueError('Need at least two distinct seed identities')
    other = np.sum(pair_weights * matrix) / pair_weights.sum()
    # In resamples, off-diagonal target-seed weights need not equal the diagonal weights.
    target_weights = pair_weights.sum(axis=0) / pair_weights.sum()
    fixed_difference_same = np.sum(sw[:, None] * q * bw) / sw.sum()
    fixed_difference_other = np.sum(target_weights[:, None] * q * bw)
    return np.array([same, other, same-other,
                     same-fixed_difference_same, other-fixed_difference_other])


NAMES = ('same_vs_B0', 'other_vs_B0', 'D_same_minus_other', 'same_vs_M1', 'other_vs_M1')


def summarize(p, q, regions, repeats=5000, random_seed=20261029):
    p, q, groups = validate(p, q, regions)
    n, b = p.shape
    point = weighted(p, q, np.ones(n), np.ones(b))
    rng = np.random.default_rng(random_seed)
    k = groups.max()+1
    draws = []
    while len(draws) < repeats:
        sw = rng.multinomial(n, np.full(n, 1/n))
        if np.count_nonzero(sw) < 2:
            continue
        gw = rng.multinomial(k, np.full(k, 1/k))
        draws.append(weighted(p, q, sw, gw[groups]))
    draws = np.asarray(draws)
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    return {name: dict(estimate=float(v), low95=float(lo), high95=float(hi))
            for name, v, lo, hi in zip(NAMES, point, lower, upper)}


def details(p, q):
    p, q, _ = validate(p, q, np.arange(np.asarray(p).shape[1]))
    choose = p > 0
    n = len(p)
    same = choose*q
    other = choose*(q.sum(axis=0, keepdims=True)-q)/(n-1)
    return dict(same_vs_B0=same, other_vs_B0=other,
                D_same_minus_other=same-other,
                same_vs_M1=same-q,
                other_vs_M1=other-(q.sum(axis=0, keepdims=True)-q)/(n-1))
