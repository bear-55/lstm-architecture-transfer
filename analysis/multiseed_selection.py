"""Leave-target-training-instance-out architecture selection (score averaging)."""
from itertools import combinations
import numpy as np


def selection_curve(p, q, sizes=(1, 3, 5, 9)):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.ndim != 2 or p.shape != q.shape or min(p.shape) < 1 or p.shape[0] < 2:
        raise ValueError('Expected matching seed x basin matrices with >=2 seeds')
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite scores must not be silently excluded')
    if not sizes or len(set(sizes)) != len(sizes) or any(type(m) is not int or not 1 <= m < len(p) for m in sizes):
        raise ValueError('Selection sizes must be distinct integers below seed count')
    out = {}
    for m in sizes:
        prob, counts = np.zeros_like(p), []
        for target in range(len(p)):
            members = np.asarray(list(combinations([s for s in range(len(p)) if s != target], m)))
            assert not (members == target).any()
            prob[target] = (p[members].mean(axis=1) > 0).mean(axis=0)
            counts.append(len(members))
        out[m] = {'selection_probability': prob, 'vs_B0': prob*q,
                  'vs_M1': -(1-prob)*q, 'subsets_per_target': counts}
    return out


def direct_curve(p, q, m):
    """Slow independent audit: compute each policy's gain before averaging."""
    n, b = p.shape
    b0, m1 = np.zeros_like(p, dtype=float), np.zeros_like(p, dtype=float)
    for t in range(n):
        sets = list(combinations([s for s in range(n) if s != t], m))
        for basin in range(b):
            for members in sets:
                choose = sum(float(p[s, basin]) for s in members)/m > 0
                b0[t, basin] += float(q[t, basin]) if choose else 0
                m1[t, basin] += 0 if choose else -float(q[t, basin])
        b0[t] /= len(sets)
        m1[t] /= len(sets)
    return b0, m1
