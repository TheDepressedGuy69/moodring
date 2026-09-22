"""Adversarial inputs. numpy warnings are errors here: a silent 0/0 becomes NaN in the output."""
import warnings

import numpy as np
import pytest

from moodring import hmm
from moodring.analysis import analyze
from moodring.render import render

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _prices(returns_pct):
    return 100 * np.exp(np.concatenate([[0], np.cumsum(np.asarray(returns_pct) / 100)]))


def _dates(n):
    return np.busday_offset(np.datetime64("2010-01-01"), np.arange(n + 1), roll="forward").astype("datetime64[D]")


def _run(x, k=3, check=True):
    c = _prices(x)
    r = analyze("X", "test", _dates(len(x)), c, k=k, check=check)
    text = render(r, color=False, width=100, height=8)
    for a in (r.filt, r.smooth, r.params.A, r.params.mu, r.params.var):
        assert np.isfinite(a).all()
    assert "nan" not in text.lower()
    return r, text


def test_constant_price():
    _run(np.zeros(900), check=False)


def test_long_halted_stretch_of_exact_zero_returns():
    x = np.random.default_rng(0).normal(0, 1, 1500)
    x[300:700] = 0.0
    r, text = _run(x)
    assert any("exactly zero" in w for w in r.warnings)


@pytest.mark.parametrize("seed", range(4))
def test_isolated_huge_outliers_never_poison_the_fit(seed):
    x = np.random.default_rng(seed).normal(0, 1, 1500)
    x[[100, 900]] = [60, -55]
    r, text = _run(x)
    assert any("25%" in w for w in r.warnings)


def test_single_crash_leaving_the_training_window():
    """A warm-started 'crisis' state loses all its data when the crash rolls out of the window."""
    x = np.random.default_rng(3).normal(0, 0.8, 3200)
    x[200:215] = np.random.default_rng(4).normal(-6, 6, 15)
    _run(x)


def test_zero_initial_probability_state_does_not_divide_by_zero():
    x = np.random.default_rng(5).normal(0, 1, 800)
    p, _ = hmm.fit(x, k=3)
    p.pi = np.array([1.0, 0.0, 0.0])
    x2 = x.copy()
    x2[0] = 12.0  # only a high-variance state can explain day one
    probs = hmm.filter_probs(x2, p)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1)


@pytest.mark.parametrize("scale", [0.01, 15.0])
def test_extreme_volatility_scales(scale):
    _run(np.random.default_rng(6).normal(0, scale, 1500))


def test_sixty_years_of_daily_data():
    _run(np.random.default_rng(1).normal(0, 1, 15000))


def test_regime_capturing_almost_no_days_is_flagged():
    x = np.random.default_rng(0).normal(0, 1, 1500)
    x[[100, 900]] = [60, -55]
    r, text = _run(x)
    assert any("covers only" in w for w in r.warnings)
    assert "!" in text
