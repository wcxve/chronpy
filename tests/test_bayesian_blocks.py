import numpy as np
import pytest
from astropy.stats import bayesian_blocks as astropy_bayesian_blocks
from numpy.testing import assert_allclose

from chronpy.hist.bayesian_blocks import blocks_binned, blocks_tte


def test_single_change_point(rseed=0):
    rng = np.random.default_rng(rseed)
    x = np.concatenate([rng.random(100), 1 + rng.random(200)])
    bb = blocks_tte(np.sort(x))
    bins = astropy_bayesian_blocks(x)
    assert len(bb.edge) == len(bins)


def test_duplicate_events(rseed=0):
    rng = np.random.default_rng(rseed)
    t = rng.random(100)
    t[80:] = t[:20]

    # Using int array as a regression test for astropy/gh-6877
    x = np.ones(t.shape, dtype=int)
    x[:20] += 1

    t_sorted = np.sort(t)
    bb1 = blocks_tte(t_sorted)
    bb2 = blocks_tte([t_sorted, t_sorted])
    bins = astropy_bayesian_blocks(t[:80], x[:80])

    assert_allclose(bb1.edge, bins)
    assert_allclose(bb2.edge, bins)


def test_zero_change_points(rseed=0):
    """
    Ensure that edges contains both endpoints when there are no change points
    """
    np.random.seed(rseed)
    # Using the failed edge case from
    # https://github.com/astropy/astropy/issues/8558
    values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2])
    bins = astropy_bayesian_blocks(values)
    bb = blocks_tte(values)
    assert_allclose(bb.edge, bins)


@pytest.fixture
def tte_data():
    rng = np.random.default_rng(0)
    t1 = np.r_[rng.uniform(-9.0, 0.5, 2000), rng.uniform(-0.5, 7.0, 2000)]
    t2 = np.r_[rng.uniform(-8.5, 1.0, 2000), rng.uniform(-0.5, 7.0, 200)]
    return np.sort(t1), np.sort(t2)


def test_tte_data(tte_data):
    t1, t2 = tte_data
    bb_joint = blocks_tte([t1, t2])
    bb = blocks_tte(np.sort(np.r_[t1, t2]))
    bins = astropy_bayesian_blocks(np.sort(np.r_[t1, t2]))
    assert_allclose(bb.edge, bins)
    assert len(bb_joint.edge) == len(bins)


def test_binned_data(tte_data):
    t1, t2 = tte_data
    tbins = np.linspace(-9, 7, 161)
    counts = np.histogram(np.sort(np.r_[t1, t2]), tbins)[0]
    bb = blocks_binned(tbins, counts)
    assert_allclose(bb.edge, [-9, -8.5, -0.5, 0.5, 1, 7])


def test_iteration(tte_data):
    t1, t2 = tte_data
    bb = blocks_tte(np.sort(np.r_[t1, t2]))
    bins = astropy_bayesian_blocks(np.sort(np.r_[t1, t2]))
    assert_allclose(bb.edge, bins)
