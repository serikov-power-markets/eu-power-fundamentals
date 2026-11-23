"""Download ENTSO-E generation unit unavailability documents (A80) for France as raw zip files, month by month with paging."""
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, r"C:\Users\user\ie-isem-market\src")
from isem import entsoe as E  # noqa: E402

FR = "10YFR-RTE------C"
OUT = Path(r"C:\Users\user\z_data\z_totsa\outages_fr")
OUT.mkdir(exist_ok=True)
for m in pd.date_range("2021-10-01", "2026-01-01", freq="MS"):
    nxt = m + pd.offsets.MonthBegin(1)
    off = 0
    while True:
        fn = OUT / f"a80_{m:%Y%m}_{off}.zip"
        if fn.exists():
            c = fn.read_bytes()
        else:
            for k in range(5):
                try:
                    c = E._get({"documentType": "A80", "biddingZone_Domain": FR, "periodStart": m.strftime("%Y%m%d0000"),
                                "periodEnd": nxt.strftime("%Y%m%d0000"), "offset": off})
                    break
                except Exception as e:
                    print("retry", m, off, str(e)[:80], flush=True)
                    time.sleep(20)
            if c is None:
                break
            fn.write_bytes(c)
        try:
            n = len(zipfile.ZipFile(io.BytesIO(c)).namelist())
        except zipfile.BadZipFile:
            n = 1
        print(m.strftime("%Y-%m"), off, n, flush=True)
        if n < 200:
            break
        off += 200
