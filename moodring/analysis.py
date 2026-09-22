from __future__ import annotations

from dataclasses import dataclass, field

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
    warnings: list = field(default_factory=list)

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

    warnings = _data_warnings(returns, smooth)
    return Report(label, source, dates[1:], close[1:], returns, params, filt, smooth, wf, disagreement, periods, warnings)


def _data_warnings(returns: np.ndarray, smooth: np.ndarray) -> list:
    """Things a careful analyst would want to know before trusting the picture."""
    out = []
    big = int((np.abs(returns) > 25.0).sum())
    if big:
        out.append(
            f"{big} daily move{'s' if big != 1 else ''} larger than 25% ({np.abs(returns).max():.0f}% at most): "
            "check for unadjusted splits or bad data, which can dominate the fit"
        )
    share = np.bincount(smooth.argmax(axis=1), minlength=smooth.shape[1]) / len(returns)
    for j, sh in enumerate(share):
        if sh < 0.01:
            out.append(
                f"regime {j + 1} of {smooth.shape[1]} covers only {sh * 100:.1f}% of days "
                f"({int(round(sh * len(returns)))}): it may be fitting a few outliers rather than a real regime; "
                "try fewer --states"
            )
    zero_share = float((returns == 0).mean())
    if zero_share > 0.05:
        out.append(f"{zero_share * 100:.0f}% of days have exactly zero return (halted or illiquid?): regimes may be unreliable")
    return out
