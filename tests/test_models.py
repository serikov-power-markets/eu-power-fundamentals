import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fund import models as M  # noqa: E402


def test_signal_pnl_arithmetic():
    idx = pd.date_range("2024-01-01", periods=4)
    spread = pd.Series([10.0, 14.0, 9.0, 9.0], index=idx)
    pred = pd.Series([0.0, 20.0, 5.0, 9.0], index=idx)
    s = M.signal_pnl(spread, pred, cost=1.0)
    # day 2: entry 10, predicted 20 -> long, realised 14 -> +4 - 1 = 3
    assert np.isclose(s.loc[idx[1], "pnl"], 3.0)
    # day 3: entry 14, predicted 5 -> short, realised 9 -> +5 - 1 = 4
    assert np.isclose(s.loc[idx[2], "pnl"], 4.0)
    # day 4: entry 9, predicted 9 -> no position, no cost
    assert s.loc[idx[3], "pnl"] == 0.0


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=5000))
    assert M.psi(a, a) < 1e-9
    assert M.psi(a, a + 2.0) > 1.0


def test_walk_forward_uses_only_the_past():
    idx = pd.date_range("2022-01-01", periods=1000)
    X = pd.DataFrame({"x": np.arange(1000.0)}, index=idx)
    y = pd.Series(np.where(idx < "2024-03-01", 1.0, 5.0), index=idx)

    class Mean:
        def fit(self, X, y):
            self.m = float(y.mean())
            return self

        def predict(self, X):
            return np.full(len(X), self.m)

    p = M.walk_forward(y, X, Mean, "2024-01-01")
    # every forecast before March 2024 is the mean of the earlier data, which is 1.0
    assert np.allclose(p["2024-01-01":"2024-02-28"].dropna(), 1.0)
    # the level shift after March is not visible until a refit after it
    assert p["2024-03-05"] < 2.0


def test_metrics_columns():
    y = pd.Series([1.0, 2.0, 3.0])
    p = pd.Series([1.0, 3.0, 5.0])
    r = M.metrics(y, p)
    assert r["mae"] == 1.0 and r["bias"] == 1.0
