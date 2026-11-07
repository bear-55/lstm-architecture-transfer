import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.multiseed_selection import selection_curve, direct_curve


def test_known_stable_heterogeneity():
    p = np.tile([.3, -.2, 0], (10, 1))
    q = np.tile([.1, -.4, -.7], (10, 1))
    for r in selection_curve(p, q).values():
        np.testing.assert_allclose(r['vs_B0'], np.tile([.1, 0, 0], (10, 1)))
        np.testing.assert_allclose(r['vs_M1'], np.tile([0, .4, .7], (10, 1)))


def test_target_early_score_never_affects_its_own_choice():
    rng = np.random.default_rng(12)
    p, q = rng.normal(size=(2, 10, 7))
    for t in range(10):
        changed = p.copy()
        changed[t] += 1e5
        a, b = selection_curve(p,q), selection_curve(changed,q)
        for m in a:
            np.testing.assert_array_equal(a[m]['vs_B0'][t], b[m]['vs_B0'][t])


def test_direct_independent_and_m1_closed_form():
    rng = np.random.default_rng(4)
    p, q = rng.normal(size=(2,10,13))
    for m,r in selection_curve(p,q).items():
        direct = direct_curve(p,q,m)
        np.testing.assert_allclose(r['vs_B0'],direct[0],atol=1e-14)
        np.testing.assert_allclose(r['vs_M1'],direct[1],atol=1e-14)
    expected = np.mean([np.mean((p[s]>0)*q[t]) for s in range(10) for t in range(10) if s!=t])
    assert selection_curve(p,q)[1]['vs_B0'].mean() == pytest.approx(expected, abs=1e-14)


def test_invalid_input_fails_instead_of_dropping_data():
    for p,q,m in [(np.ones((1,2)),np.ones((1,2)),(1,)),
                  (np.ones((3,2)),np.ones((3,2)),(3,)),
                  (np.full((3,2),np.nan),np.ones((3,2)),(1,))]:
        with pytest.raises(ValueError):
            selection_curve(p,q,m)
