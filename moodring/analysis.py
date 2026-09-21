from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import hmm
from .walkforward import WalkForwardResult, hindsight_disagreement, walk_forward

MIN_RETURNS = 250


@dataclass
class Report:
    label: str
    source: str
    dates: np.ndarray  # (T,) date of each return
    close: np.ndarray  # (T,) close on that date
    returns: np.ndarray  # (T,) log returns in percent
    params: hmm.HMMParams
    filt: np.ndarray  # (T, K) causal probabilities
    smooth: np.ndarray  # (T, K) hindsight probabilities
    wf: WalkForwardResult | None
    disagreement: float | None
    periods: float = 252.0

    @property
    def k(self) -> int:
        return self.params.k


def analyze(
    label: str,
    source: str,
    dates: np.ndarray,
    close: np.ndarray,
    k: int = 3,
    check: bool = True,
    refit_every: int = 126,
    min_train: int = 504,
    progress=None,
) -> Report:
    returns = 100.0 * np.diff(np.log(close))
    if len(returns) < MIN_RETURNS:
        raise ValueError(f"need at least {MIN_RETURNS} daily returns, got {len(returns)}")
    years = max((dates[-1] - dates[0]).astype(float) / 365.25, 1e-9)
    periods = float(round(len(returns) / years))
    params, _ = hmm.fit(returns, k=k, n_init=3)
    filt = hmm.filter_probs(returns, params)
    smooth = hmm.smooth_probs(returns, params)

    wf = None
    disagreement = None
    if check and len(returns) >= min_train + refit_every:
        wf = walk_forward(returns, k=k, min_train=min_train, refit_every=refit_every, periods=periods, progress=progress)
        disagreement = hindsight_disagreement(smooth.argmax(axis=1), wf)

    return Report(label, source, dates[1:], close[1:], returns, params, filt, smooth, wf, disagreement, periods)
