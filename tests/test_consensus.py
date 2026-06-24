"""Deterministic tests for orchestration/consensus.py (ADR-0004 b2.3).

Strategy: build N references that share a known LOOK but have different CONTENT, confirm the
consensus recovers the look and that an outlier reference doesn't dominate (median robustness)."""

from __future__ import annotations

import numpy as np

from lutgen.orchestration.consensus import ConsensusLook, build_consensus
from lutgen.orchestration.stats import compute_stats

WARM = np.array([0.15, 0.0, -0.1])   # the shared "look": warm cast


def _ref(seed: int, bias=WARM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    content = rng.random((48, 48, 3)) * 0.6 + 0.2   # varied content
    return np.clip(content + bias, 0.0, 1.0)


def _stats_set(seeds, bias=WARM):
    return [compute_stats(_ref(s, bias)) for s in seeds]


def test_consensus_recovers_shared_look():
    c = build_consensus(_stats_set(range(6)))
    assert isinstance(c, ConsensusLook)
    assert c.n_refs == 6
    # warm look present in every band of the consensus
    assert np.all(c.band_balance[:, 0] > c.band_balance[:, 2])


def test_low_variance_traits_are_high_confidence():
    # Same look, varied content → look-driven traits should be consistent (high confidence).
    c = build_consensus(_stats_set(range(8)))
    assert c.confidence["balance"] > 0.9
    assert all(0.0 < w <= 1.0 for w in c.confidence.values())


def test_outlier_does_not_dominate():
    warm = _stats_set(range(5))
    clean = build_consensus(warm)
    # inject one strongly COLD outlier
    outlier = compute_stats(_ref(99, bias=np.array([-0.2, 0.0, 0.25])))
    polluted = build_consensus(warm + [outlier])
    # median stays warm; balance shifts only a little
    assert np.all(polluted.band_balance[:, 0] > polluted.band_balance[:, 2])
    shift = np.abs(polluted.band_balance - clean.band_balance).max()
    assert shift < 0.05


def test_deterministic():
    a = build_consensus(_stats_set(range(4)))
    b = build_consensus(_stats_set(range(4)))
    np.testing.assert_array_equal(a.band_balance, b.band_balance)
    np.testing.assert_array_equal(a.channel_quantiles, b.channel_quantiles)


def test_requires_stats():
    import pytest

    with pytest.raises(ValueError):
        build_consensus([])
