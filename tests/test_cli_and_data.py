import numpy as np
import pytest

from moodring.analysis import analyze
from moodring.cli import main
from moodring.data import DataError, fetch_yahoo, load_csv
from moodring.render import render
from moodring.synth import demo_prices


def test_load_csv_prefers_adj_close_and_sorts(tmp_path):
    f = tmp_path / "p.csv"
    lines = ["Date,Open,Close,Adj Close"]
    for i in range(60):
        d = np.datetime64("2020-01-01") + np.timedelta64(i, "D")
        lines.append(f"{d},1,{100 + i},{50 + i}")
    lines = [lines[0]] + lines[1:][::-1]  # reversed order
    f.write_text("\n".join(lines))
    dates, close = load_csv(str(f))
    assert (np.diff(dates.astype(int)) > 0).all()
    assert close[0] == 50.0


def test_load_csv_bad_header(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("foo,bar\n1,2\n")
    with pytest.raises(DataError, match="Date"):
        load_csv(str(f))


def test_load_csv_missing_file():
    with pytest.raises(DataError):
        load_csv("/no/such/file.csv")


def test_symbol_validation_blocks_url_injection():
    with pytest.raises(DataError):
        fetch_yahoo("SPY/../../x?y=1")


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("color", [True, False])
def test_render_all_modes(k, color):
    dates, close = demo_prices(1500)
    r = analyze("DEMO", "synthetic", dates, close, k=k, check=True)
    text = render(r, color=color, width=100, height=10)
    assert "REALITY CHECK" in text
    assert ("\033[" in text) == color


def test_periods_inferred_for_24_7_assets():
    n = 900
    dates = np.datetime64("2020-01-01") + np.arange(n + 1).astype("timedelta64[D]")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, n + 1)))
    r = analyze("BTC", "test", dates, close, k=2, check=False)
    assert r.periods == 365


def test_cli_demo_runs(capsys):
    assert main(["--demo", "--no-color", "--width", "90", "--no-check"]) == 0
    out = capsys.readouterr().out
    assert "moodring" in out and "As of" in out


def test_cli_requires_input(capsys):
    with pytest.raises(SystemExit):
        main([])
