"""Download public Census/USGS display boundaries; reruns reuse local files."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
CENSUS = "https://tigerweb.geo.census.gov/arcgis/rest/services/Generalized_ACS2024/State_County/MapServer/8"
USGS = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/1"
JOBS = [
    ("census_states_5m_2024.geojson", CENSUS,
     dict(where="STATE NOT IN ('02','15','60','66','69','72','78')",
          outFields="STATE,NAME", outSR=4326, returnGeometry="true",
          geometryPrecision=5, f="geojson"), 49),
    ("usgs_huc2_regions.geojson", USGS,
     dict(where="huc2 >= '01' AND huc2 <= '18'", outFields="huc2,name",
          outSR=4326, returnGeometry="true", maxAllowableOffset=0.01,
          geometryPrecision=5, f="geojson"), 18),
]


def fetch(job):
    name, base, params, expected = job
    url = base + "/query?" + urlencode(params)
    path = OUT / name
    if not path.exists():
        with urlopen(url, timeout=45) as response:
            payload = response.read()
        data = json.loads(payload)
        if "error" in data:
            raise RuntimeError(data["error"])
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == expected
        assert not data.get("exceededTransferLimit", False)
        path.write_bytes(payload)
    payload = path.read_bytes()
    data = json.loads(payload)
    assert len(data["features"]) == expected
    metadata_path = OUT / name.replace(".geojson", "_metadata.json")
    if not metadata_path.exists():
        with urlopen(base + "?f=pjson", timeout=45) as response:
            metadata_path.write_bytes(response.read())
    print(name, len(data["features"]), len(payload), flush=True)
    return dict(file=name, url=url, metadata_url=base,
                downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                feature_count=len(data["features"]), bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                crs="EPSG:4326 (longitude, latitude)", parameters=params)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, JOBS))
    (OUT / "geography_sources.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
