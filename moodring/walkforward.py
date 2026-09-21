"""Out-of-sample honesty check for regime models.

Everything here only ever uses information that existed at the time:
parameters are fit on a past window, then frozen while the regime is *filtered*
(not smoothed) through the next block of days. Then we ask a plain question:
does knowing the regime today help predict tomorrow's volatility better than a
trailing 21-day realized-volatility number that costs nothing to compute?
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import hmm

HORIZON = 5  # days ahead for the forecast-quality comparison
BASELINE_WINDOW = 21


@dataclass
class WalkForwardResult:
    start: int  # first out-of-sample index
    probs: np.ndarray  # (T, K), NaN before `start`
    pred_var: np.ndarray  # (T,), NaN before `start`: forecast variance for the next day
    labels: np.ndarray  # (T,), -1 before `start`
    n_refits: int
    metrics: dict = field(default_factory=dict)


def walk_forward(
    x: np.ndarray,
    k: int = 3,
    min_train: int = 504,
    refit_every: int = 126,
    max_train: int = 1260,
    periods: float = 252.0,
    progress=None,
) -> WalkForwardResult:
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < min_train + refit_every:
        raise ValueError(
            f"need at least {min_train + refit_every} daily returns for a walk-forward check, got {T}"
        )
    probs = np.full((T, k), np.nan)
    pred_var = np.full(T, np.nan)
    labels = np.full(T, -1, dtype=int)

    starts = list(range(min_train, T, refit_every))
    params = None
    for i, t0 in enumerate(starts):
        lo = max(0, t0 - max_train)
        params, _ = hmm.fit(x[lo:t0], k=k, n_init=2, init=None if params is None else params)
        t1 = min(T, t0 + refit_every)
        # filter through the recent past so the state estimate is warmed up, using frozen params
        warm = max(0, t0 - 250)
        f = hmm.filter_probs(x[warm:t1], params)[t0 - warm:]
        probs[t0:t1] = f
        pred_var[t0:t1] = (f @ params.A) @ params.var
        labels[t0:t1] = f.argmax(axis=1)
        if progress:
            progress(i + 1, len(starts))

    res = WalkForwardResult(min_train, probs, pred_var, labels, len(starts))
    res.metrics = _metrics(x, res, k, periods)
    return res


def _ranks(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(float)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _trailing_var(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    sq = x**2
    csum = np.cumsum(np.concatenate([[0.0], sq]))
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def _forward_realized(x: np.ndarray, h: int) -> np.ndarray:
    """Mean squared return over days t+1..t+h (NaN where the window runs off the end)."""
    T = len(x)
    out = np.full(T, np.nan)
    sq = x**2
    csum = np.cumsum(np.concatenate([[0.0], sq]))
    for t in range(T - h):
        out[t] = (csum[t + 1 + h] - csum[t + 1]) / h
    return out


def _metrics(x: np.ndarray, res: WalkForwardResult, k: int, periods: float) -> dict:
    T = len(x)
    s = res.start
    fwd1 = np.full(T, np.nan)
    fwd1[:-1] = x[1:] ** 2
    fwd_h = _forward_realized(x, HORIZON)

    idx = np.arange(s, T - HORIZON)
    base_var = _trailing_var(x, BASELINE_WINDOW)

    hmm_corr = _spearman(res.pred_var[idx], fwd_h[idx])
    base_corr = _spearman(base_var[idx], fwd_h[idx])

    # next-day realized vol by regime label (annualized RMS return, in percent)
    by_regime = []
    lab = res.labels
    for j in range(k):
        sel = idx[lab[idx] == j]
        if len(sel) >= 5:
            by_regime.append(
                {"regime": j, "days": int(len(sel)), "ann_vol": float(np.sqrt(fwd1[sel].mean() * periods))}
            )
        else:
            by_regime.append({"regime": j, "days": int(len(sel)), "ann_vol": float("nan")})

    # baseline: k buckets of trailing vol, cut points from the *past only*
    bucket = np.full(T, -1, dtype=int)
    for t in idx:
        past = base_var[BASELINE_WINDOW - 1:t + 1]
        if len(past) < 50:
            continue
        cuts = np.quantile(past, np.linspace(0, 1, k + 1)[1:-1])
        bucket[t] = int(np.searchsorted(cuts, base_var[t]))
    base_by = []
    for j in range(k):
        sel = idx[bucket[idx] == j]
        base_by.append(
            {"bucket": j, "days": int(len(sel)),
             "ann_vol": float(np.sqrt(fwd1[sel].mean() * periods)) if len(sel) >= 5 else float("nan")}
        )

    return {
        "n_oos_days": int(len(idx)),
        "horizon": HORIZON,
        "spearman_hmm": hmm_corr,
        "spearman_baseline": base_corr,
        "by_regime": by_regime,
        "baseline_by_bucket": base_by,
    }


def hindsight_disagreement(smoothed_labels: np.ndarray, wf: WalkForwardResult) -> float:
    """Share of out-of-sample days where the hindsight label differs from the real-time one."""
    sel = np.arange(wf.start, len(smoothed_labels))
    return float((smoothed_labels[sel] != wf.labels[sel]).mean())
