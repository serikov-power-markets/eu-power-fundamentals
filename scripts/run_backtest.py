"""Day-ahead price forecasts for DE-LU and FR, the DE-FR spread, a stylised signal and monitoring, walk-forward 2024-2025.

Fuel costs: TTF front month (daily) and EUA auction prices end in December 2025, so the study period is 2022 to 2025."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fund import data as D  # noqa: E402
from fund import features as F  # noqa: E402
from fund import models as M  # noqa: E402

CACHE = Path.home() / "z_data" / "z_totsa" / "cache"
HOME = Path.home()
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
END = "2025-12-31"
de = F.zone_frame(CACHE, "DE", "FR").loc[:END]
fr = F.zone_frame(CACHE, "FR", "DE").loc[:END]
fuel = F.fuel_costs(HOME / "eu-gas-storage-valuation" / "data" / "raw" / "ttf_front_month.json", HOME / "ie-isem-market" / "results" / "eua.pkl")
flows = pd.concat([F.daily(D.flow(CACHE, "DE", "FR")), F.daily(D.flow(CACHE, "FR", "DE"))], axis=1)
flows.columns = ["flow_de_fr", "flow_fr_de"]
flows["net_de_to_fr"] = flows["flow_de_fr"] - flows["flow_fr_de"]
# the physical flow series has gaps on 638 of 1,761 days (no data from the platform), so it is not used as a feature
flows = flows[[]]
X = M.build(de, fr, fuel, flows)
OUTF = RES / "fr_nuclear_unavailability_known_dminus1.csv"
HAVE_OUT = OUTF.exists()
if HAVE_OUT:
    o = pd.read_csv(OUTF, index_col=0, parse_dates=True)
    X["fr_out_planned"], X["fr_out_forced"], X["fr_units_out"], X["fr_planned_next"] = o["ongoing_planned_mw"], o["ongoing_forced_mw"], o["ongoing_units"], o["planned_start_next_mw"]
START = "2024-01-01"
out = {"period": [str(X.index.min().date()), END], "test_start": START}

BASE = ["de_load_fc", "de_wind_fc", "de_solar_fc", "de_resid_fc", "fr_load_fc", "fr_wind_fc", "fr_solar_fc", "fr_resid_fc", "gas_cost_l2", "ttf_l2", "eua_l2",
        "fr_nuclear_l2", "fr_nuclear_m7_l2", "fr_hydro_res_l2", "de_gas_l2", "de_coal_lignite_l2", "s1", "c1", "dow"]
LAGS = ["de_price_l1", "de_price_l7", "de_price_m7", "fr_price_l1", "fr_price_l7", "fr_price_m7"]
STACK = ["de_resid_fc", "fr_resid_fc", "gas_cost_l2", "eua_l2", "fr_nuclear_m7_l2"]
tab = []
preds = {}
for tgt, zone in (("price_DE", "de"), ("price_FR", "fr"), ("spread", None)):
    y = de["price"] if tgt == "price_DE" else fr["price"] if tgt == "price_FR" else de["price"] - fr["price"]
    own = f"{zone}_price_l1" if zone else "de_price_l1"
    models = {
        "last delivery day": None,
        "stack: ridge on residual load, fuel cost and nuclear": lambda: M.Cols(STACK, M.ridge()),
        "ridge: stack and all fundamentals and lags": lambda: M.Cols(BASE + LAGS, M.ridge()),
        "delta: stack plus gradient boosting of its residual": lambda: M.Delta(STACK, BASE + LAGS),
        "gradient boosting: all fundamentals and lags": lambda: M.Cols(BASE + LAGS, M.gbm()),
        "gradient boosting: fundamentals only, no price lags": lambda: M.Cols(BASE, M.gbm()),
    }
    if HAVE_OUT:
        OUTC = ["fr_out_planned", "fr_out_forced", "fr_units_out", "fr_planned_next"]
        models["delta: as above plus French nuclear outages known on D-1"] = lambda: M.Delta(STACK + ["fr_out_planned", "fr_out_forced"], BASE + LAGS + OUTC)
        models["gradient boosting: all plus outages"] = lambda: M.Cols(BASE + LAGS + OUTC, M.gbm())
    P = {}
    for name, mk in models.items():
        if mk is None:
            P[name] = (X["de_price_l1"] - X["fr_price_l1"]) if tgt == "spread" else X[own]
        else:
            P[name] = M.walk_forward(y, X, mk, START)
    P = pd.DataFrame(P)
    preds[tgt] = P
    m = (y.index >= START) & P.notna().all(axis=1) & y.notna()
    for name in P.columns:
        r = M.metrics(y[m], P.loc[m, name])
        r.update(target=tgt, model=name)
        tab.append(r)
t = pd.DataFrame(tab)[["target", "model", "days", "mae", "rmse", "bias", "corr"]]
t.to_csv(RES / "forecast_metrics.csv", index=False)
out["forecast_metrics"] = t.to_dict("records")
print(t.to_string())

# direction of the change of the spread against the previous delivery day (no tradable instrument at that price: a measure of information,
# not of profit). Benchmark rules: revert to the 7-day mean spread; repeat the previous change.
y = de["price"] - fr["price"]
prev = y.shift(1)
cand = {name: preds["spread"][name] for name in preds["spread"].columns if name != "last delivery day"}
cand["rule: revert to the 7-day mean spread"] = (X["de_price_m7"] - X["fr_price_m7"])
cand["rule: repeat the previous change"] = prev + (prev - y.shift(2))
rows = []
for name, p in cand.items():
    m = (y.index >= START) & p.notna() & prev.notna() & y.notna()
    pos = np.sign(p[m] - prev[m])
    move = (y[m] - prev[m])
    nz = move != 0
    rows.append({"predictor": name, "days": int(m.sum()), "hit_rate_direction": round(float((np.sign(move[nz]) == pos[nz]).mean()), 3),
                 "mean_pnl_vs_previous_day_eur_mwh_cost0.5": round(float((pos * move - 0.5 * (pos != 0)).mean()), 2),
                 "mean_abs_daily_change": round(float(move.abs().mean()), 2)})
sg = pd.DataFrame(rows).sort_values("hit_rate_direction", ascending=False)
sg.to_csv(RES / "spread_direction.csv", index=False)
best_name = t[t.target == "spread"].sort_values("mae").iloc[0]["model"]
out["spread_direction"] = sg.to_dict("records")
print(sg.to_string())
# paired comparison of the direction hits with and without the outage features (exact McNemar test)
from scipy.stats import binomtest  # noqa: E402
a_name, b_name = "delta: as above plus French nuclear outages known on D-1", "delta: stack plus gradient boosting of its residual"
if a_name in cand:
    m = (y.index >= START) & cand[a_name].notna() & cand[b_name].notna() & prev.notna() & y.notna() & ((y - prev) != 0)
    mv = (y - prev)[m]
    ha = np.sign(cand[a_name][m] - prev[m]) == np.sign(mv)
    hb = np.sign(cand[b_name][m] - prev[m]) == np.sign(mv)
    n10, n01 = int((ha & ~hb).sum()), int((~ha & hb).sum())
    out["outage_effect_on_direction"] = {"only_with_outages_right": n10, "only_without_outages_right": n01,
                                         "p_exact_mcnemar": round(float(binomtest(n10, n10 + n01, 0.5).pvalue), 4) if n10 + n01 else None}
    print(out["outage_effect_on_direction"])

# monitoring: rolling 30-day MAE of the best spread model against its first test year, PSI of key features 2024 vs 2025
p = preds["spread"][best_name]
err = (p - y).abs()
roll = err.rolling(30).mean()
base_mae = float(err[(err.index >= START) & (err.index < "2025-01-01")].mean())
alerts = roll[(roll > 1.5 * base_mae) & (roll.index >= "2025-01-01")]
mon = {"mae_2024": round(base_mae, 2), "mae_2025": round(float(err[err.index >= "2025-01-01"].mean()), 2),
       "days_rolling30_above_1.5x_2024_mae": int(len(alerts))}
tr = X[(X.index < "2024-01-01")]
te = X[(X.index >= "2025-01-01")]
mon["psi_2025_vs_pre2024"] = {c: round(M.psi(tr[c], te[c]), 2) for c in ("de_resid_fc", "fr_resid_fc", "gas_cost_l2", "eua_l2", "fr_nuclear_m7_l2", "de_price_l1")}
out["monitoring"] = mon
roll.rename("rolling30_mae").to_csv(RES / "rolling_mae_spread.csv")
(RES / "summary.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("monitoring",)}, indent=1, default=str))
