"""A small Gaussian hidden Markov model, written from scratch on numpy.

Why not hmmlearn? Its ``predict_proba`` returns *smoothed* probabilities, which
use the whole sample (including the future) to label each day. That is fine for
describing history and useless for trading. Here the causal ``filter`` is the
first-class citizen: ``filter(x[:t])`` never changes when data after ``t`` is
appended.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_FLOOR = 1e-6


@dataclass
class HMMParams:
    pi: np.ndarray  # (K,) initial state distribution
    A: np.ndarray  # (K, K) row-stochastic transition matrix
    mu: np.ndarray  # (K,) state means
    var: np.ndarray  # (K,) state variances

    @property
    def k(self) -> int:
        return len(self.pi)

    def sorted_by_variance(self) -> "HMMParams":
        order = np.argsort(self.var)
        return HMMParams(
            self.pi[order], self.A[np.ix_(order, order)], self.mu[order], self.var[order]
        )

    def n_free_params(self) -> int:
        k = self.k
        return (k - 1) + k * (k - 1) + 2 * k


def _log_emission(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2 * np.pi * var)[None, :] + (x[:, None] - mu[None, :]) ** 2 / var[None, :])


def _forward(logB: np.ndarray, pi: np.ndarray, A: np.ndarray):
    T, K = logB.shape
    shift = logB.max(axis=1, keepdims=True)
    B = np.exp(logB - shift)
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = np.maximum(pi, 1e-12) * B[0]  # a state's initial probability may have decayed to exactly 0
    c[0] = max(a.sum(), 1e-300)
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        s = max(a.sum(), 1e-300)
        c[t] = s
        alpha[t] = a / s
    loglik = float(np.log(c).sum() + shift.sum())
    return alpha, c, B, loglik


def _backward(B: np.ndarray, c: np.ndarray, A: np.ndarray) -> np.ndarray:
    T, K = B.shape
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


def _quantile_init(x: np.ndarray, k: int, rng: np.random.Generator, jitter: bool) -> HMMParams:
    kernel = np.ones(5) / 5
    s = np.convolve(np.abs(x), kernel, mode="same")
    if jitter:
        s = s + rng.normal(0.0, 0.3 * s.std() + 1e-12, size=len(s))
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    groups = np.searchsorted(edges, s)
    mu = np.empty(k)
    var = np.empty(k)
    overall = x.var() + 1e-12
    for j in range(k):
        g = x[groups == j]
        if len(g) < 3:
            g = x
        mu[j] = g.mean()
        var[j] = max(g.var(), 1e-3 * overall)
    A = np.full((k, k), 0.05 / max(k - 1, 1))
    np.fill_diagonal(A, 0.95)
    return HMMParams(np.full(k, 1.0 / k), A, mu, var)


def _em(x: np.ndarray, p: HMMParams, max_iter: int, tol: float):
    T = len(x)
    var_floor = 1e-3 * (x.var() + 1e-12)
    pi, A, mu, var = p.pi.copy(), p.A.copy(), p.mu.copy(), p.var.copy()
    prev = -np.inf
    ll = prev
    for _ in range(max_iter):
        logB = _log_emission(x, mu, var)
        alpha, c, B, ll = _forward(logB, pi, A)
        beta = _backward(B, c, A)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi_sum = A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))

        pi = np.maximum(gamma[0], A_FLOOR)
        pi = pi / pi.sum()
        # A state that owns no data (e.g. a warm-started "crisis" state whose crisis has left the
        # window) must keep its old parameters, or 0/0 turns the whole fit into NaN.
        rows = xi_sum.sum(axis=1, keepdims=True)
        live_rows = rows[:, 0] > 1e-9
        A_new = A.copy()
        A_new[live_rows] = xi_sum[live_rows] / rows[live_rows]
        A = np.maximum(A_new, A_FLOOR)
        A /= A.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        live = w > 1e-6
        mu_new = mu.copy()
        var_new = var.copy()
        mu_new[live] = (gamma[:, live] * x[:, None]).sum(axis=0) / w[live]
        var_new[live] = np.maximum(
            (gamma[:, live] * (x[:, None] - mu_new[None, live]) ** 2).sum(axis=0) / w[live], var_floor
        )
        mu, var = mu_new, var_new

        if abs(ll - prev) / T < tol:
            break
        prev = ll
    return HMMParams(pi, A, mu, var), ll


def fit(
    x: np.ndarray,
    k: int = 3,
    n_init: int = 2,
    max_iter: int = 150,
    tol: float = 1e-6,
    seed: int = 0,
    init: HMMParams | None = None,
) -> tuple[HMMParams, float]:
    """Fit by EM. Returns params sorted by ascending variance, and the log-likelihood."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    starts = [init] if init is not None else [
        _quantile_init(x, k, rng, jitter=(i > 0)) for i in range(max(1, n_init))
    ]
    best = None
    for start in starts:
        params, ll = _em(x, start, max_iter, tol)
        if best is None or ll > best[1]:
            best = (params, ll)
    return best[0].sorted_by_variance(), best[1]


def filter_probs(x: np.ndarray, p: HMMParams) -> np.ndarray:
    """Causal state probabilities P(s_t | x_1..x_t). Row t uses data up to and including t only."""
    x = np.asarray(x, dtype=float)
    alpha, _, _, _ = _forward(_log_emission(x, p.mu, p.var), p.pi, p.A)
    return alpha


def smooth_probs(x: np.ndarray, p: HMMParams) -> np.ndarray:
    """Hindsight state probabilities P(s_t | x_1..x_T). Uses the future; for description only."""
    x = np.asarray(x, dtype=float)
    alpha, c, B, _ = _forward(_log_emission(x, p.mu, p.var), p.pi, p.A)
    gamma = alpha * _backward(B, c, p.A)
    return gamma / gamma.sum(axis=1, keepdims=True)


def loglik(x: np.ndarray, p: HMMParams) -> float:
    x = np.asarray(x, dtype=float)
    return _forward(_log_emission(x, p.mu, p.var), p.pi, p.A)[3]


def bic(x: np.ndarray, p: HMMParams) -> float:
    return -2.0 * loglik(x, p) + p.n_free_params() * np.log(len(x))
