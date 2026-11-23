import sys, time
from pathlib import Path
# ENTSO-E helper of the sibling project ie-isem-market (token from ENTSOE_TOKEN or ~/.entsoe_token, never written to disk)
sys.path.insert(0, str(Path.home() / "ie-isem-market" / "src"))
from isem import entsoe as E
C = Path.home() / "z_data" / "z_totsa" / "cache"
DE, FR = "10Y1001A1001A82H", "10YFR-RTE------C"
S, T = "2022-01-01", "2026-10-01"
jobs = []
for n, z in (("DE", DE), ("FR", FR)):
    jobs += [
     (f"price_{n}", {"documentType": "A44", "in_Domain": z, "out_Domain": z}, "price.amount"),
     (f"load_actual_{n}", {"documentType": "A65", "processType": "A16", "outBiddingZone_Domain": z}, "quantity"),
     (f"load_fc_{n}", {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": z}, "quantity"),
     (f"res_fc_{n}", {"documentType": "A69", "processType": "A01", "in_Domain": z}, "quantity"),
     (f"gen_{n}", {"documentType": "A75", "processType": "A16", "in_Domain": z}, "quantity"),
    ]
jobs += [("flow_DE_FR", {"documentType": "A11", "in_Domain": FR, "out_Domain": DE}, "quantity"),
         ("flow_FR_DE", {"documentType": "A11", "in_Domain": DE, "out_Domain": FR}, "quantity")]
for name, p, vt in jobs:
    for k in range(5):
        try:
            df = E.fetch(name, p, S, T, C, vt); print(name, len(df), flush=True); break
        except Exception as e:
            print("fail", name, type(e).__name__, str(e)[:100], flush=True); time.sleep(30)
