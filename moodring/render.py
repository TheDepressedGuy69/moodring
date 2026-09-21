"""Terminal report: regime-shaded price chart, regime table, and the reality check."""
from __future__ import annotations

import numpy as np

from .analysis import Report

NAMES = {
    2: ["calm", "turbulent"],
    3: ["calm", "choppy", "turbulent"],
    4: ["calm", "normal", "choppy", "turbulent"],
}
COLORS = {
    2: [71, 203],
    3: [71, 179, 203],
    4: [71, 149, 209, 197],
}
SHADES = ["░", "▒", "▓", "█"]
EIGHTHS = "▁▂▃▄▅▆▇"
MARGIN = 10


class Style:
    def __init__(self, color: bool):
        self.color = color

    def paint(self, text: str, code: int | None = None, bold: bool = False, dim: bool = False) -> str:
        if not self.color:
            return text
        parts = []
        if bold:
            parts.append("1")
        if dim:
            parts.append("2")
        if code is not None:
            parts.append(f"38;5;{code}")
        if not parts:
            return text
        return f"\033[{';'.join(parts)}m{text}\033[0m"


def _price(p: float) -> str:
    return f"{p:,.0f}" if p >= 1000 else f"{p:.2f}"


def _bins(n: int, width: int) -> np.ndarray:
    width = max(1, min(width, n))
    return np.linspace(0, n, width + 1).astype(int)


def _mode_labels(labels: np.ndarray, edges: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(len(edges) - 1, dtype=int)
    for i in range(len(out)):
        seg = labels[edges[i]:max(edges[i + 1], edges[i] + 1)]
        out[i] = int(np.bincount(seg, minlength=k).argmax())
    return out


def _chart(r: Report, st: Style, width: int, height: int) -> list[str]:
    k = r.k
    T = len(r.close)
    edges = _bins(T, width - MARGIN - 1)
    cols = len(edges) - 1
    logp = np.log(r.close)
    v = np.array([logp[edges[i]:max(edges[i + 1], edges[i] + 1)].mean() for i in range(cols)])
    lo, hi = v.min(), v.max()
    span = (hi - lo) or 1.0
    level = (np.round((v - lo) / span * (height * 8 - 1)).astype(int)) + 1
    live = _mode_labels(r.filt.argmax(axis=1), edges, k)
    hind = _mode_labels(r.smooth.argmax(axis=1), edges, k)
    codes = COLORS[k]

    lines = []
    for row in range(height - 1, -1, -1):
        if row == height - 1:
            margin = f"{_price(r.close.max()):>{MARGIN - 1}} "
        elif row == 0:
            margin = f"{_price(r.close.min()):>{MARGIN - 1}} "
        else:
            margin = " " * MARGIN
        cells = []
        for c in range(cols):
            fill = level[c] - row * 8
            ch = "█" if fill >= 8 else (EIGHTHS[fill - 1] if fill > 0 else " ")
            cells.append(st.paint(ch, codes[live[c]]) if ch != " " else " ")
        lines.append(st.paint(margin, dim=True) + "".join(cells))

    def strip(name: str, labels: np.ndarray) -> str:
        cells = [
            st.paint("▄", codes[j]) if st.color else SHADES[min(j, 3)]
            for j in labels
        ]
        return st.paint(f"{name:>{MARGIN - 1}} ", dim=True) + "".join(cells)

    lines.append("")
    lines.append(strip("live", live))
    lines.append(strip("hindsight", hind))

    axis = [" "] * (cols + MARGIN)
    years = r.dates[edges[:-1]].astype("datetime64[Y]").astype(int) + 1970
    last_end = -99
    for c in range(cols):
        if c == 0 or years[c] != years[c - 1]:
            label = str(years[c])
            pos = MARGIN + c
            if pos > last_end + 2 and pos + 4 <= len(axis):
                axis[pos:pos + 4] = list(label)
                last_end = pos + 4
    lines.append(st.paint("".join(axis), dim=True))
    return lines


def _bar(p: float, st: Style, code: int, n: int = 10) -> str:
    filled = int(round(p * n))
    return st.paint("█" * filled, code) + st.paint("·" * (n - filled), dim=True)


def render(r: Report, color: bool = True, width: int = 100, height: int = 12) -> str:
    st = Style(color)
    k = r.k
    names, codes = NAMES[k], COLORS[k]
    p = r.params
    out: list[str] = []

    span = f"{r.dates[0]} → {r.dates[-1]}"
    out.append(
        st.paint(f" moodring · {r.label}", bold=True)
        + st.paint(f"  {span} · {len(r.returns):,} days · {k} regimes · {r.source}", dim=True)
    )
    out.append("")
    out.extend(_chart(r, st, width, height))
    legend = "  ".join(st.paint("█ " if color else f"{SHADES[min(j, 3)]} ", codes[j]) + names[j] for j in range(k))
    out.append(" " * MARGIN + legend)
    out.append("")

    hard = r.smooth.argmax(axis=1)
    share = np.bincount(hard, minlength=k) / len(hard)
    now = r.filt[-1]
    out.append(st.paint(" regime      share   ann.vol   ann.ret   avg stay   right now", bold=True))
    for j in range(k):
        stay = 1.0 / max(1e-9, 1.0 - p.A[j, j])
        row = (
            f" {names[j]:<10} {share[j] * 100:5.1f}%  {np.sqrt(p.var[j] * r.periods):6.1f}%  "
            f"{p.mu[j] * r.periods:+7.1f}%  {stay:6.0f} d   "
            f"{_bar(now[j], st, codes[j])} {now[j] * 100:3.0f}%"
        )
        out.append(st.paint(row[:12], codes[j], bold=True) + row[12:])
    out.append("")

    cur = int(now.argmax())
    nxt = now @ p.A
    out.append(
        f" As of {r.dates[-1]}: "
        + st.paint(names[cur].upper(), codes[cur], bold=True)
        + f" ({now[cur] * 100:.0f}% confident, using data up to today only)"
    )
    out.append(
        " Tomorrow's odds: "
        + "  ".join(f"{names[j]} {nxt[j] * 100:.0f}%" for j in range(k))
    )
    out.append("")

    out.append(st.paint(" transitions (row = today, column = tomorrow)", bold=True))
    out.append("            " + "".join(f"{n[:9]:>10}" for n in names))
    for j in range(k):
        out.append(
            f" {names[j]:<10} " + "".join(f"{p.A[j, m] * 100:9.1f}%" for m in range(k))
        )
    out.append("")

    out.extend(_reality_check(r, st, names, codes))
    out.append("")
    out.append(st.paint(" Research tool, not investment advice. Regimes describe volatility, not direction.", dim=True))
    return "\n".join(out)


def _reality_check(r: Report, st: Style, names: list, codes: list) -> list[str]:
    if r.wf is None:
        return [st.paint(" reality check skipped (--no-check, or too little history)", dim=True)]
    m = r.wf.metrics
    start = r.dates[r.wf.start]
    out = [
        st.paint(" REALITY CHECK", bold=True)
        + st.paint(
            f"  walk-forward: {r.wf.n_refits} refits, out-of-sample from {start} ({m['n_oos_days']:,} days), "
            "only past data used",
            dim=True,
        ),
        "",
        " Volatility of the day AFTER each real-time regime call (annualised):",
    ]
    vols = []
    for entry in m["by_regime"]:
        j = entry["regime"]
        v = entry["ann_vol"]
        vols.append(v)
        vs = "   n/a" if not np.isfinite(v) else f"{v:5.1f}%"
        out.append(f"   {st.paint(f'{names[j]:<10}', codes[j])} {vs}   ({entry['days']:,} days)")
    finite = [v for v in vols if np.isfinite(v)]
    ordered = len(finite) == len(vols) and all(a < b for a, b in zip(finite, finite[1:]))
    weakest = min((b / a for a, b in zip(finite, finite[1:]) if a > 0), default=1.0)
    if not ordered:
        note = "regimes do NOT cleanly separate future volatility out of sample"
    elif weakest < 1.15:
        note = "regimes only barely separate future volatility (two are near-identical)"
    else:
        note = "regimes separate future volatility in the right order"
    out.append("   " + st.paint(note, dim=True))
    out.append("")

    h, b = m["spearman_hmm"], m["spearman_baseline"]
    out.append(f" Predicting {m['horizon']}-day-ahead variance (rank correlation, higher is better):")
    out.append(f"   HMM regime forecast      {h:5.2f}")
    out.append(f"   trailing 21-day vol      {b:5.2f}   <- the free baseline")
    diff = h - b
    if diff > 0.05:
        verdict = "the HMM adds real information beyond trailing volatility"
    elif diff > 0.02:
        verdict = "a modest edge over trailing volatility"
    elif diff >= -0.02:
        verdict = "a wash: about as good as trailing volatility, not better"
    else:
        verdict = "trailing volatility is the better forecaster here"
    out.append("   " + st.paint(verdict, bold=True))
    out.append(st.paint("   (one sample, no significance test: treat gaps under ~0.05 as noise)", dim=True))
    out.append("")
    if r.disagreement is not None:
        out.append(
            f" Hindsight gap: {r.disagreement * 100:.0f}% of out-of-sample days would have been labelled "
            "differently in real time."
        )
        out.append(st.paint("   (compare the 'live' and 'hindsight' strips above)", dim=True))
    return out
