from __future__ import annotations

import argparse
import os
import shutil
import sys

from . import __version__
from .analysis import analyze
from .data import DataError, fetch_yahoo, load_csv
from .render import render
from .synth import demo_prices


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="moodring",
        description="A mood ring for the market: detect volatility regimes with a hidden Markov model, "
        "then check honestly whether they'd have helped in real time.",
    )
    p.add_argument("symbol", nargs="?", help="ticker symbol, e.g. SPY, QQQ, BTC-USD, ^VIX")
    p.add_argument("--csv", metavar="FILE", help="read prices from a CSV with Date and Close/Adj Close columns")
    p.add_argument("--demo", action="store_true", help="use built-in synthetic data (works offline)")
    p.add_argument("--years", type=float, default=10.0, help="years of history to fetch (default: 10)")
    p.add_argument("--states", type=int, default=3, choices=(2, 3, 4), help="number of regimes (default: 3)")
    p.add_argument("--no-check", action="store_true", help="skip the walk-forward reality check")
    p.add_argument("--refit", type=int, default=126, metavar="DAYS", help="walk-forward refit interval (default: 126)")
    p.add_argument("--min-train", type=int, default=504, metavar="DAYS", help="walk-forward warm-up window (default: 504)")
    p.add_argument("--width", type=int, default=None, help="output width (default: terminal width)")
    p.add_argument("--height", type=int, default=12, help="chart height in rows (default: 12)")
    p.add_argument("--no-color", action="store_true", help="plain output (also honours NO_COLOR)")
    p.add_argument("--version", action="version", version=f"moodring {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (args.symbol or args.csv or args.demo):
        parser.error("give a ticker (moodring SPY), --csv FILE, or --demo")

    try:
        if args.demo:
            dates, close = demo_prices()
            label, source = "DEMO", "synthetic data"
        elif args.csv:
            dates, close = load_csv(args.csv)
            label, source = os.path.basename(args.csv), "csv"
        else:
            symbol = args.symbol.upper()
            print(f"fetching {symbol} from Yahoo Finance...", file=sys.stderr)
            dates, close = fetch_yahoo(symbol, args.years)
            label, source = symbol, "Yahoo Finance (adj. close)"
    except DataError as exc:
        print(f"moodring: {exc}", file=sys.stderr)
        return 1

    tty_err = sys.stderr.isatty()

    def progress(i: int, n: int) -> None:
        if tty_err:
            print(f"\rwalk-forward check {i}/{n}", end="", file=sys.stderr, flush=True)

    try:
        report = analyze(
            label, source, dates, close,
            k=args.states, check=not args.no_check,
            refit_every=args.refit, min_train=args.min_train, progress=progress,
        )
    except ValueError as exc:
        print(f"moodring: {exc}", file=sys.stderr)
        return 1
    if tty_err:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    color = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    width = args.width or shutil.get_terminal_size((100, 30)).columns
    width = max(60, min(width, 140))
    print(render(report, color=color, width=width, height=max(6, args.height)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
