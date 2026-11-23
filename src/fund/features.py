"""Daily features known the day before delivery, for Germany-Luxembourg (DE) and France (FR).

Information set at 10:00 on D-1, before the day-ahead auction: day-ahead wind, solar and load forecasts for day D (ASSUMED to be
published before the auction), day-ahead prices of delivery day D-1 and earlier (published on D-2), actual generation, load and
cross-border flows of D-1 and earlier is NOT assumed (they are published with a delay), so those enter lagged by two days, the
TTF front month close of D-2 and the latest EUA auction price at least two days old.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

CCGT_EFF = 0.5
CCGT_EF = 0.202      # tCO2 per MWh of gas burned


def daily(s: pd.Series | pd.DataFrame, how="mean"):
    return getattr(s.resample("1D"), how)()


def fuel_costs(ttf_json: Path, eua_pkl: Path) -> pd.DataFrame:
    import json
    j = json.load(open(ttf_json))["chart"]["result"][0]
    ttf = pd.Series(j["indicators"]["quote"][0]["close"], index=pd.to_datetime(j["timestamp"], unit="s").normalize(), name="ttf").dropna()
    ttf = ttf[~ttf.index.duplicated(keep="last")]
    eua = pd.read_pickle(eua_pkl).drop_duplicates("date").set_index("date")["price"].rename("eua")
    idx = pd.date_range("2021-12-01", "2026-10-31", freq="D")
    f = pd.concat([ttf.reindex(idx).ffill(), eua.reindex(idx).ffill()], axis=1)
    f["gas_cost"] = f["ttf"] / CCGT_EFF + CCGT_EF / CCGT_EFF * f["eua"]
    return f


def zone_frame(cache: Path, zone: str, other: str) -> pd.DataFrame:
    p = D.price(cache, zone)
    out = pd.DataFrame({"price": daily(p)})
    out["price_peak"] = daily(p[(p.index.hour >= 8) & (p.index.hour < 20)])
    lf = D.load_actual(cache, zone)               # actual load (published with a delay): used lagged
    out["load_act"] = daily(lf)
    fc = D.res_forecast(cache, zone)
    wind = [c for c in fc.columns if "wind" in c]
    solar = [c for c in fc.columns if "solar" in c]
    out["wind_fc"] = daily(fc[wind].sum(axis=1, min_count=1)) if wind else np.nan
    out["solar_fc"] = daily(fc[solar].sum(axis=1, min_count=1)) if solar else np.nan
    g = D.generation(cache, zone)
    if f"gen_nuclear_{zone}" in g.columns:
        out["nuclear"] = daily(g[f"gen_nuclear_{zone}"])
    for k in ("hydro_res", "hydro_ror", "gas", "coal", "B02", "solar", "wind_onshore", "wind_offshore"):
        if f"gen_{k}_{zone}" in g.columns:
            out["act_" + k] = daily(g[f"gen_{k}_{zone}"])
    lfc = D.read(cache, f"load_fc_{zone}")
    lfc["ts"] = pd.to_datetime(lfc["ts_utc"], utc=True).dt.tz_convert(None)
    out["load_fc"] = daily(lfc.drop_duplicates("ts").set_index("ts")["value"].resample("1h").mean())
    return out
