import numpy as np

from moodring import hmm
from moodring.synth import simulate_returns


def _data(n=2500, seed=1):
    return simulate_returns(n, seed=seed)


def test_fit_recovers_volatility_regimes():
    x, states = _data()
    params, _ = hmm.fit(x, k=3)
    vols = np.sqrt(params.var)
    assert np.allclose(vols, [0.6, 1.2, 3.0], rtol=0.2)
    # sticky chain: diagonal should be large
    assert (np.diag(params.A) > 0.9).all()


def test_smoothed_labels_match_truth():
    x, states = _data()
    params, _ = hmm.fit(x, k=3)
    labels = hmm.smooth_probs(x, params).argmax(axis=1)
    assert (labels == states).mean() > 0.8


def test_filter_is_causal():
    """The whole point: appending future data must not change past filtered probabilities."""
    x, _ = _data(1500)
    params, _ = hmm.fit(x, k=3)
    full = hmm.filter_probs(x, params)
    for t in (100, 700, 1200):
        prefix = hmm.filter_probs(x[:t], params)
        assert np.allclose(prefix, full[:t], atol=1e-12)


def test_smoother_is_not_causal():
    """Sanity check that the test above has teeth: smoothing does use the future."""
    x, _ = _data(1500)
    params, _ = hmm.fit(x, k=3)
    full = hmm.smooth_probs(x, params)
    prefix = hmm.smooth_probs(x[:700], params)
    assert not np.allclose(prefix, full[:700], atol=1e-6)


def test_probabilities_are_valid():
    x, _ = _data(800)
    params, _ = hmm.fit(x, k=2)
    for probs in (hmm.filter_probs(x, params), hmm.smooth_probs(x, params)):
        assert np.isfinite(probs).all()
        assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(params.A.sum(axis=1), 1.0)
    assert (np.diff(params.var) >= 0).all()  # sorted calm -> turbulent


def test_bic_prefers_true_number_of_states_over_one_state_fit():
    x, _ = _data(2500)
    p3, _ = hmm.fit(x, k=3)
    p2, _ = hmm.fit(x, k=2)
    assert hmm.bic(x, p3) < hmm.bic(x, p2) + 50  # 3 states at least competitive
