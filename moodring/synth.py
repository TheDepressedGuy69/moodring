"""Synthetic regime-switching price series, for --demo and for tests."""
from __future__ import annotations

import numpy as np

DEFAULT_VOLS = (0.6, 1.2, 3.0)  # daily vol in percent: calm, choppy, turbulent
DEFAULT_MEANS = (0.06, 0.02, -0.12)  # daily mean in percent


def simulate_returns(
    n: int,
    seed: int = 0,
    vols: tuple = DEFAULT_VOLS,
    means: tuple = DEFAULT_MEANS,
    stay: float = 0.985,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (returns_in_percent, true_states) from a sticky Markov chain."""
    rng = np.random.default_rng(seed)
    k = len(vols)
    A = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(A, stay)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(k, p=A[states[t - 1]])
    vols_a, means_a = np.asarray(vols), np.asarray(means)
    x = rng.normal(means_a[states], vols_a[states])
    return x, states


def demo_prices(n: int = 2520, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return (dates, close) for a synthetic ~10 year daily series."""
    x, _ = simulate_returns(n, seed=seed)
    close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(x / 100.0)]))
    dates = np.busday_offset(np.datetime64("2015-01-01"), np.arange(n + 1), roll="forward")
    return dates.astype("datetime64[D]"), close
