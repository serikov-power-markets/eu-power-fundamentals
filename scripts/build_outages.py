"""Parse the downloaded A80 documents for France and write daily nuclear unavailability known at 10:00 on D-1."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fund import outages as O  # noqa: E402

folder = Path.home() / "z_data" / "z_totsa" / "outages_fr"
df = O.read_folder(folder)
print("revisions", len(df), "documents", df["doc"].nunique(), "nuclear rows", int((df.psr == "B14").sum()))
days = pd.date_range("2022-01-01", "2025-12-31", freq="D")
k = O.ongoing_at_gate(df, days)
k.to_csv(ROOT / "results" / "fr_nuclear_unavailability_known_dminus1.csv", index_label="day")
print(k.describe().round(0))
