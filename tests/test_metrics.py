"""Hand-calculated checks for yield / DPMO / sigma level and the two-proportion test."""
import pytest

from src.metrics import ctq_metrics, two_proportion_test


def test_dpmo_and_sigma_level_hand():
    # 66 fails in 1,000 runs: p = 0.066, DPMO = 66,000, yield = 93.4%
    m = ctq_metrics(1000, 66)
    assert m["dpmo"] == pytest.approx(66_000)
    assert m["yield"] == pytest.approx(0.934)


def test_sigma_level_textbook_value():
    # A 2.275% defect rate is the area beyond Z = 2 on a standard normal curve
    # (1 - Phi(2) = 0.02275), so long-term Z = 2.0 and sigma level = 2.0 + 1.5 = 3.5.
    m = ctq_metrics(100_000, 2275)
    assert m["z_long_term"] == pytest.approx(2.0, abs=1e-3)
    assert m["sigma_level"] == pytest.approx(3.5, abs=1e-3)


def test_classic_six_sigma_dpmo():
    # 3.4 DPMO is the famous "six sigma": Z_long_term = 4.5, plus 1.5 shift = 6.0
    m = ctq_metrics(10_000_000, 34)
    assert m["sigma_level"] == pytest.approx(6.0, abs=0.01)


def test_two_proportion_hand():
    # 20/100 vs 10/100: p_pool = 0.15; SE = sqrt(0.15 * 0.85 * (1/100 + 1/100)) = 0.050498
    # z = (0.20 - 0.10) / 0.050498 = 1.9803; two-sided p = 2 * (1 - Phi(1.9803)) = 0.04767
    t = two_proportion_test(20, 100, 10, 100)
    assert t["z"] == pytest.approx(1.9803, abs=1e-4)
    assert t["p_value"] == pytest.approx(0.04767, abs=1e-4)
    # CI: 0.10 +/- 1.96 * sqrt(0.2*0.8/100 + 0.1*0.9/100) = 0.10 +/- 1.96 * 0.05 = (0.002, 0.198)
    assert t["ci_low"] == pytest.approx(0.002, abs=1e-3)
    assert t["ci_high"] == pytest.approx(0.198, abs=1e-3)
