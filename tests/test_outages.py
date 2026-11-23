import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fund import outages as O  # noqa: E402

XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<Unavailability_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:outagedocument:3:0">
  <mRID>DOC1</mRID><revisionNumber>2</revisionNumber><type>A80</type>
  <createdDateTime>2024-03-01T08:00:00Z</createdDateTime>
  <docStatus><value>A05</value></docStatus>
  <TimeSeries><mRID>1</mRID><businessType>A54</businessType>
    <production_RegisteredResource.mRID codingScheme="A01">U1</production_RegisteredResource.mRID>
    <production_RegisteredResource.name>UNIT ONE</production_RegisteredResource.name>
    <production_RegisteredResource.pSRType.psrType>B14</production_RegisteredResource.pSRType.psrType>
    <production_RegisteredResource.pSRType.powerSystemResources.nominalP unit="MAW">900.0</production_RegisteredResource.pSRType.powerSystemResources.nominalP>
    <Available_Period><timeInterval><start>2024-03-05T00:00Z</start><end>2024-03-07T00:00Z</end></timeInterval>
      <resolution>PT1M</resolution><Point><position>1</position><quantity>300</quantity></Point></Available_Period>
  </TimeSeries>
</Unavailability_MarketDocument>"""


def _df():
    rows = O.parse_document(XML)
    r2 = dict(rows[0])
    r2.update(rev=1, created="2024-02-01T08:00:00Z", avail=900.0)        # earlier revision: no loss yet
    df = pd.DataFrame(rows + [r2])
    for c in ("created", "start", "end"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(None)
    return df


def test_parse_reads_unit_and_capacity():
    r = O.parse_document(XML)[0]
    assert r["unit"] == "U1" and r["psr"] == "B14" and r["business"] == "A54" and r["nominal"] == 900.0 and r["avail"] == 300.0


def test_gate_uses_only_revisions_created_before_the_decision():
    days = pd.DatetimeIndex(["2024-03-05", "2024-03-06", "2024-03-08"])
    k = O.known_unavailable(_df(), days)
    assert np.isclose(k.loc["2024-03-05", "forced_mw"], 600.0)          # revision 2 was created before 10:00 on 4 March
    assert np.isclose(k.loc["2024-03-06", "forced_mw"], 600.0)
    assert k.loc["2024-03-08", "forced_mw"] == 0.0                      # outside the period
    # revision 2 created after the gate of 5 March (4 March 10:00) is not seen: only revision 1, without loss
    late = _df()
    late.loc[late.rev == 2, "created"] = pd.Timestamp("2024-03-04T12:00")
    k2 = O.known_unavailable(late, pd.DatetimeIndex(["2024-03-05", "2024-03-06"]))
    assert k2.loc["2024-03-05", "forced_mw"] == 0.0
    assert np.isclose(k2.loc["2024-03-06", "forced_mw"], 600.0)         # by the gate of 6 March it has been published


def test_cancelled_documents_are_ignored():
    df = _df()
    df["status"] = "A09"
    k = O.known_unavailable(df, pd.DatetimeIndex(["2024-03-05"]))
    assert k.loc["2024-03-05", "forced_mw"] == 0.0


def test_ongoing_at_gate_ignores_end_date_and_creation_time():
    df = _df()
    df = df[df.rev == 2].copy()                                  # final revision only, created long after the outage began
    df["created"] = pd.Timestamp("2024-03-20")                   # created at the end, as in the real data
    k = O.ongoing_at_gate(df, pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-08"]))
    # outage 5 March 00:00 to 7 March 00:00, forced
    assert k.loc["2024-03-04", "ongoing_forced_mw"] == 0.0       # gate 3 March 10:00, not started
    assert np.isclose(k.loc["2024-03-06", "ongoing_forced_mw"], 600.0)   # gate 5 March 10:00, ongoing
    assert np.isclose(k.loc["2024-03-05", "ongoing_forced_mw"], 0.0)     # gate 4 March 10:00: not started (forced start is not announced)
    assert k.loc["2024-03-08", "ongoing_forced_mw"] == 0.0       # gate 7 March 10:00: ended
