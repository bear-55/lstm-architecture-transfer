"""Use the same budget estimates and empirical fits as Figure 5."""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    (HERE/"data").mkdir(exist_ok=True)
    for name in ("curve_endpoints.csv", "power_fits.csv"):
        shutil.copyfile(HERE.parent/"Fig05"/"data"/name, HERE/"data"/name)
