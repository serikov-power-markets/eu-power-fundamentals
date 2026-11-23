"""Fundamental baseline, delta model, walk-forward evaluation, stylised signal and monitoring for daily day-ahead prices."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build(de: pd.DataFrame, fr: pd.DataFrame, fuel: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """One row per delivery day D with features known at 10:00 on D-1 (see features.py for the information set)."""
    X = pd.DataFrame(index=de.index)
    for z, f in (("de", de), ("fr", fr)):
        X[f"{z}_load_fc"] = f["load_fc"]
        X[f"{z}_wind_fc"] = f["wind_fc"]
        X[f"{z}_solar_fc"] = f["solar_fc"]
        X[f"{z}_resid_fc"] = f["load_fc"] - f["wind_fc"] - f["solar_fc"]
        X[f"{z}_price_l1"] = f["price"].shift(1)            # delivery day D-1, published on D-2
        X[f"{z}_price_l7"] = f["price"].shift(7)
        X[f"{z}_price_m7"] = f["price"].shift(1).rolling(7).mean()
    X["fr_nuclear_l2"] = fr["nuclear"].shift(2)
    X["fr_nuclear_m7_l2"] = fr["nuclear"].shift(2).rolling(7).mean()
    X["fr_hydro_res_l2"] = fr["act_hydro_res"].shift(2)
    X["de_gas_l2"] = de["act_gas"].shift(2)
    X["de_coal_lignite_l2"] = (de["act_coal"] + de["act_B02"]).shift(2)
    X["gas_cost_l2"] = fuel["gas_cost"].reindex(X.index).shift(2)
    X["ttf_l2"] = fuel["ttf"].reindex(X.index).shift(2)
    X["eua_l2"] = fuel["eua"].reindex(X.index).shift(2)
    for c in flows.columns:
        X[c + "_l2"] = flows[c].reindex(X.index).shift(2)
    doy = X.index.dayofyear
    X["s1"], X["c1"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["dow"] = X.index.dayofweek
    return X


STACK = ["de_resid_fc", "fr_resid_fc", "gas_cost_l2", "eua_l2", "fr_nuclear_m7_l2"]


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))


def gbm(seed=0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=seed)


class Delta:
    """Stack baseline (ridge on a few fundamentals) plus a gradient boosting model of its residual (the delta model)."""
    def __init__(self, stack_cols, delta_cols):
        self.sc, self.dc = stack_cols, delta_cols

    def fit(self, X, y):
        self.base = ridge().fit(X[self.sc], y)
        res = y - self.base.predict(X[self.sc])
        self.delta = gbm().fit(X[self.dc], res)
        return self

    def predict(self, X):
        return self.base.predict(X[self.sc]) + self.delta.predict(X[self.dc])


class Cols:
    def __init__(self, cols, model):
        self.cols, self.m = cols, model

    def fit(self, X, y):
        self.m.fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.m.predict(X[self.cols])


def walk_forward(y: pd.Series, X: pd.DataFrame, make_model, start: str, refit_freq="MS") -> pd.Series:
    """Expanding window, refitted at each month start; days with missing features are skipped."""
    pred = pd.Series(np.nan, index=y.index)
    ok = X.notna().all(axis=1).values & y.notna().values
    starts = pd.date_range(start, y.index.max(), freq=refit_freq)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else y.index.max() + pd.Timedelta(days=1)
        tr = (X.index < s) & ok
        te = (X.index >= s) & (X.index < e) & X.notna().all(axis=1).values
        if tr.sum() < 250 or te.sum() == 0:
            continue
        m = make_model().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def metrics(y: pd.Series, p: pd.Series) -> dict:
    m = y.notna() & p.notna()
    e = (p - y)[m]
    return {"days": int(m.sum()), "mae": round(float(e.abs().mean()), 2), "rmse": round(float(np.sqrt((e ** 2).mean())), 2),
            "bias": round(float(e.mean()), 2), "corr": round(float(np.corrcoef(p[m], y[m])[0, 1]), 3)}


def signal_pnl(spread: pd.Series, pred: pd.Series, cost: float = 0.5) -> pd.DataFrame:
    """Stylised spread trade. Entry price = the spread of the previous delivery day (ASSUMED stand-in for a forward, there is no
    tradable instrument in the data); position = sign of (predicted spread - entry); P&L per MWh = position x (realised - entry) - cost."""
    entry = spread.shift(1)
    pos = np.sign(pred - entry)
    pnl = pos * (spread - entry) - cost * (pos != 0)
    return pd.DataFrame({"entry": entry, "pred": pred, "realised": spread, "pos": pos, "pnl": pnl}).dropna()


def psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    """Population stability index of a feature between the training window and a later window."""
    q = np.unique(np.quantile(train.dropna(), np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return 0.0
    a = np.histogram(train.dropna(), q)[0] / max(train.notna().sum(), 1)
    b = np.histogram(test.dropna(), q)[0] / max(test.notna().sum(), 1)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(((b - a) * np.log(b / a)).sum())
