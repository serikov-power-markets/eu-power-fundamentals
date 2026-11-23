"""ENTSO-E generation unit unavailability documents (A80): parsing and the unavailable capacity known the day before delivery.

A document has revisions; each revision (created at some time) replaces the previous one of the same document. For delivery day D
the information set at 10:00 on D-1 is the latest revision created by then, ignoring cancelled documents. The unavailable capacity of a
unit on day D is nominal power minus the available capacity of the period, weighted by the share of the day the period covers.
Resolution of the quantity series inside a period is not unpacked: the mean of the points is used when a period has several points.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

NS = re.compile(r"^\{.*?\}")


def _tag(e):
    return NS.sub("", e.tag)


def _find(e, name):
    for c in e.iter():
        if _tag(c) == name:
            return c
    return None


def parse_document(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    top = {_tag(c): (c.text or "").strip() for c in root if len(c) == 0}
    status = _find(root, "docStatus")
    st = _find(status, "value").text.strip() if status is not None and _find(status, "value") is not None else ""
    rows = []
    for ts in (c for c in root if _tag(c) == "TimeSeries"):
        d = {_tag(c): (c.text or "").strip() for c in ts if len(c) == 0}
        nominal = _find(ts, "production_RegisteredResource.pSRType.powerSystemResources.nominalP")
        psr = _find(ts, "production_RegisteredResource.pSRType.psrType")
        unit = _find(ts, "production_RegisteredResource.mRID")
        name = _find(ts, "production_RegisteredResource.name")
        for ap in (c for c in ts if _tag(c) == "Available_Period"):
            iv = _find(ap, "timeInterval")
            s, e = _find(iv, "start").text, _find(iv, "end").text
            q = [float(p.text) for p in ap.iter() if _tag(p) == "quantity"]
            rows.append({"doc": top.get("mRID"), "rev": int(top.get("revisionNumber", 0)), "created": top.get("createdDateTime"), "status": st,
                         "business": d.get("businessType"), "unit": unit.text.strip() if unit is not None else None,
                         "name": name.text.strip() if name is not None else None, "psr": psr.text.strip() if psr is not None else None,
                         "nominal": float(nominal.text) if nominal is not None else np.nan, "start": s, "end": e,
                         "avail": float(np.mean(q)) if q else np.nan})
    return rows


def read_folder(folder: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(folder).glob("a80_*.zip")):
        try:
            z = zipfile.ZipFile(io.BytesIO(f.read_bytes()))
        except zipfile.BadZipFile:
            continue
        for n in z.namelist():
            rows += parse_document(z.read(n))
    df = pd.DataFrame(rows).drop_duplicates(["doc", "rev", "unit", "start", "end"])
    for c in ("created", "start", "end"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(None)
    return df


def current_revisions(df: pd.DataFrame) -> pd.DataFrame:
    """Each revision with the time from which it is valid (its creation) and until which (the creation of the next revision of the document)."""
    d = df.sort_values(["doc", "rev"]).copy()
    d["valid_from"] = d["created"]
    d["valid_to"] = d.groupby(["doc", "unit"])["created"].shift(-1).fillna(pd.Timestamp("2100-01-01"))
    return d


def known_unavailable(df: pd.DataFrame, days: pd.DatetimeIndex, psr: str = "B14", gate_hour: int = 10) -> pd.DataFrame:
    """Unavailable capacity (MW, day average) of units of type `psr` for each delivery day, as known at gate_hour on D-1,
    split into planned (A53) and forced (A54) outages, plus the number of units with an outage."""
    d = current_revisions(df)
    d = d[(d["psr"] == psr) & (d["status"] != "A09")].copy()
    d["lost"] = (d["nominal"] - d["avail"]).clip(lower=0)
    out = []
    for day in days:
        gate = day - pd.Timedelta(days=1) + pd.Timedelta(hours=gate_hour)
        k = d[(d.valid_from <= gate) & (d.valid_to > gate)]
        ov = (k["end"].clip(upper=day + pd.Timedelta(days=1)) - k["start"].clip(lower=day)).dt.total_seconds().clip(lower=0) / 86400.0
        k = k.assign(w=ov * k["lost"])
        k = k[k.w > 0]
        # one outage per unit and type at a time: take the largest loss per unit
        by_unit = k.groupby(["unit", "business"])["w"].max().unstack("business").reindex(columns=["A53", "A54"]).fillna(0.0)
        out.append({"planned_mw": float(by_unit["A53"].sum()), "forced_mw": float(by_unit["A54"].sum()), "units_out": int((by_unit.sum(axis=1) > 0).sum())})
    return pd.DataFrame(out, index=days)


def ongoing_at_gate(df: pd.DataFrame, days: pd.DatetimeIndex, psr: str = "B14", gate_hour: int = 10) -> pd.DataFrame:
    """Point-in-time safe version. The API returns only the final revision of each document, created when the outage ends, so the
    creation time cannot be used as the publication time (see `known_unavailable`, which therefore misses long outages). Here the
    information set at the gate is: outages that have started and not ended at the gate (they are assumed to continue through day D,
    the end date is not used) and planned outages (A53) that start between the gate and the end of day D (assumed announced in advance).
    The data hold the realised start and end, so a start that was only known later is the remaining leak (stated in the README)."""
    d = df[(df["psr"] == psr) & (df["status"] != "A09")].copy()
    d["lost"] = (d["nominal"] - d["avail"]).clip(lower=0)
    rows = []
    for day in days:
        gate = day - pd.Timedelta(days=1) + pd.Timedelta(hours=gate_hour)
        on = d[(d.start <= gate) & (d.end > gate)]
        by = on.groupby(["unit", "business"])["lost"].max().unstack("business").reindex(columns=["A53", "A54"]).fillna(0.0)
        nxt = d[(d.business == "A53") & (d.start > gate) & (d.start < day + pd.Timedelta(days=1))]
        rows.append({"ongoing_planned_mw": float(by["A53"].sum()), "ongoing_forced_mw": float(by["A54"].sum()),
                     "ongoing_units": int((by.sum(axis=1) > 0).sum()), "planned_start_next_mw": float(nxt.groupby("unit")["lost"].max().sum())})
    return pd.DataFrame(rows, index=days)
