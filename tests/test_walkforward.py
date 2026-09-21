import numpy as np
import pytest

from moodring.synth import simulate_returns
from moodring.walkforward import walk_forward


def test_no_lookahead_leakage():
    """Out-of-sample output for the first T1 days must be identical whether or not
    later data exists. If any future information leaked in, these would differ."""
    x, _ = simulate_returns(1500, seed=3)
    short = walk_forward(x[:1100], k=3, min_train=504, refit_every=126)
    full = walk_forward(x, k=3, min_train=504, refit_every=126)
    assert np.allclose(short.probs[:1100], full.probs[:1100], equal_nan=True, atol=1e-10)
    assert np.array_equal(short.labels[:1100], full.labels[:1100])


def test_out_of_sample_starts_after_warmup():
    x, _ = simulate_returns(1400, seed=4)
    wf = walk_forward(x, k=3, min_train=504, refit_every=126)
    assert (wf.labels[:504] == -1).all()
    assert (wf.labels[504:] >= 0).all()
    assert np.isnan(wf.probs[:504]).all()
    assert np.isfinite(wf.probs[504:]).all()


def test_regimes_separate_future_volatility_on_regime_data():
    x, _ = simulate_returns(3000, seed=5)
    wf = walk_forward(x, k=3, min_train=504, refit_every=126)
    vols = [e["ann_vol"] for e in wf.metrics["by_regime"]]
    assert vols[0] < vols[1] < vols[2]
    assert wf.metrics["spearman_hmm"] > 0.3


def test_too_little_data_is_a_clear_error():
    x, _ = simulate_returns(300, seed=6)
    with pytest.raises(ValueError, match="at least"):
        walk_forward(x, min_train=504, refit_every=126)
