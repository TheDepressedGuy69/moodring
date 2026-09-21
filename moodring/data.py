"""Price loading: Yahoo Finance's public chart endpoint (stdlib only), or a CSV file."""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

SYMBOL_RE = re.compile(r"^[A-Za-z0-9.^=\-]{1,20}$")
USER_AGENT = "Mozilla/5.0 (compatible; moodring)"


class DataError(RuntimeError):
    pass


def _clean(dates: list, close: list) -> tuple[np.ndarray, np.ndarray]:
    d = np.array(dates, dtype="datetime64[D]")
    c = np.array(close, dtype=float)
    keep = np.isfinite(c) & (c > 0)
    d, c = d[keep], c[keep]
    order = np.argsort(d, kind="stable")
    d, c = d[order], c[order]
    _, first = np.unique(d, return_index=True)  # drop duplicate dates
    return d[first], c[first]


def fetch_yahoo(symbol: str, years: float = 10.0, timeout: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    if not SYMBOL_RE.match(symbol):
        raise DataError(f"'{symbol}' doesn't look like a ticker symbol")
    now = int(time.time())
    start = int(now - years * 365.25 * 86400)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
        f"?period1={start}&period2={now}&interval=1d&events=div%2Csplit"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise DataError(f"Yahoo Finance has no data for '{symbol}'") from exc
        raise DataError(f"Yahoo Finance returned HTTP {exc.code}; try --csv instead") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DataError(f"couldn't reach Yahoo Finance ({exc}); try --csv or --demo") from exc

    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        desc = (chart.get("error") or {}).get("description", "no data returned")
        raise DataError(f"Yahoo Finance: {desc}")
    result = chart["result"][0]
    stamps = result.get("timestamp") or []
    ind = result.get("indicators", {})
    series = None
    if ind.get("adjclose"):
        series = ind["adjclose"][0].get("adjclose")
    if series is None and ind.get("quote"):
        series = ind["quote"][0].get("close")
    if not stamps or series is None:
        raise DataError(f"Yahoo Finance returned no prices for '{symbol}'")
    offset = int(result.get("meta", {}).get("gmtoffset", 0))
    dates = [(int(t) + offset) // 86400 for t in stamps]
    close = [np.nan if v is None else v for v in series]
    dates = [np.datetime64(int(d), "D") for d in dates]
    return _clean(dates, close)


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        f = open(path, newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise DataError(f"can't read {path}: {exc.strerror}") from exc
    with f:
        reader = csv.reader(f)
        try:
            header = [h.strip().lower() for h in next(reader)]
        except StopIteration:
            raise DataError(f"{path} is empty") from None
        date_col = next((i for i, h in enumerate(header) if h in ("date", "datetime", "time", "timestamp")), None)
        price_col = next(
            (i for name in ("adj close", "adj_close", "adjclose", "close", "price")
             for i, h in enumerate(header) if h == name),
            None,
        )
        if date_col is None or price_col is None:
            raise DataError(f"{path}: need a 'Date' column and a 'Close' (or 'Adj Close') column, found {header}")
        dates, close = [], []
        for row in reader:
            try:
                dates.append(np.datetime64(row[date_col].strip()[:10], "D"))
                close.append(float(row[price_col]))
            except (ValueError, IndexError):
                continue
    if len(dates) < 30:
        raise DataError(f"{path}: only {len(dates)} usable rows")
    return _clean(dates, close)
