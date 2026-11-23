"""Loaders for the cached ENTSO-E tables (see scripts/download.py): hourly UTC series for Germany-Luxembourg and France."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _hourly(df: pd.DataFrame, how: str) -> pd.Series:
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts_utc"], utc=True).dt.tz_convert(None)
    s = d.groupby("ts")["value"].sum() if "series" in d.columns and d["series"].nunique() > 1 else d.groupby("ts")["value"].mean()
    return s.resample("1h").mean() if how == "mean" else s.resample("1h").mean()


def read(cache: Path, name: str) -> pd.DataFrame:
    fs = sorted(Path(cache).glob(f"{name}_*.parquet"))
    if not fs:
        raise FileNotFoundError(name)
    return pd.read_parquet(fs[-1])


def price(cache: Path, zone: str) -> pd.Series:
    """Day-ahead price in EUR/MWh, hourly mean (the day-ahead market has 15-minute products from October 2025)."""
    d = read(cache, f"price_{zone}")
    d["ts"] = pd.to_datetime(d["ts_utc"], utc=True).dt.tz_convert(None)
    return d.drop_duplicates("ts").set_index("ts")["value"].resample("1h").mean().rename(f"price_{zone}")


def load_actual(cache: Path, zone: str) -> pd.Series:
    d = read(cache, f"load_actual_{zone}")
    d["ts"] = pd.to_datetime(d["ts_utc"], utc=True).dt.tz_convert(None)
    return d.drop_duplicates("ts").set_index("ts")["value"].resample("1h").mean().rename(f"load_{zone}")


def res_forecast(cache: Path, zone: str) -> pd.DataFrame:
    """Day-ahead wind and solar forecast (MW, hourly mean) by series."""
    d = read(cache, f"res_fc_{zone}")
    d["ts"] = pd.to_datetime(d["ts_utc"], utc=True).dt.tz_convert(None)
    p = d.groupby(["ts", "series"])["value"].mean().unstack("series")
    p.columns = [f"fc_{c}_{zone}" for c in p.columns]
    return p.resample("1h").mean()


def generation(cache: Path, zone: str) -> pd.DataFrame:
    """Actual generation by production type (MW, hourly mean)."""
    d = read(cache, f"gen_{zone}")
    d["ts"] = pd.to_datetime(d["ts_utc"], utc=True).dt.tz_convert(None)
    p = d.groupby(["ts", "series"])["value"].mean().unstack("series")
    p.columns = [f"gen_{c}_{zone}" for c in p.columns]
    return p.resample("1h").mean()


def flow(cache: Path, a: str, b: str) -> pd.Series:
    d = read(cache, f"flow_{a}_{b}")
    d["ts"] = pd.to_datetime(d["ts_utc"], utc=True).dt.tz_convert(None)
    return d.drop_duplicates("ts").set_index("ts")["value"].resample("1h").mean().rename(f"flow_{a}_{b}")
