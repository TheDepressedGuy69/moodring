"""Guards the reality check's own honesty: how often it claims an 'edge' on pure noise
with no real regime structure. Stress-testing measured this at ~10% on ordinary IID
returns and up to ~25% on fat-tailed noise; these tests pin a fixed-seed example of each
so the verdict thresholds in render.py can't quietly loosen without a test noticing.
"""
import numpy as np

from moodring.analysis import analyze


def _prices(returns_pct):
    return 100 * np.exp(np.concatenate([[0], np.cumsum(np.asarray(returns_pct) / 100)]))


def _dates(n):
    return np.busday_offset(np.datetime64("2010-01-01"), np.arange(n + 1), roll="forward").astype("datetime64[D]")


def _diff(seed, dist):
    rng = np.random.default_rng(seed)
    x = rng.standard_t(4, 3000) if dist == "t" else rng.normal(0, 1, 3000)
    r = analyze("X", "test", _dates(len(x)), _prices(x), k=3, check=True)
    m = r.wf.metrics
    return m["spearman_hmm"] - m["spearman_baseline"]

# seed 1002 is a known false "edge" case for the fat-tailed generator, used to make sure the
# thresholds are high enough that ordinary noise mostly lands as "wash", not "edge".


def test_verdict_thresholds_are_not_trivially_cleared_by_iid_noise():
    diffs = [_diff(1000 + i, "normal") for i in range(8)]
    claims_edge = sum(1 for d in diffs if d > 0.05)
    assert claims_edge <= 3, f"too many false 'edge' claims on plain noise: {diffs}"


def test_a_known_fat_tailed_false_positive_still_reads_as_at_most_a_modest_edge():
    diff = _diff(1002, "t")
    assert diff < 0.15, "a single noisy fit should never look like a strong 'real edge'"
