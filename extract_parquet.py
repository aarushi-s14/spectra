"""
Extract spectral features from parquet image datasets (GenImage BigGAN = fake,
Open Images = real). Writes features.json merged with any prior rows.
"""
import sys, os, io, json
import numpy as np
import cv2
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_core import SpectralDetector

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
LIMIT = 4000


def feats_from_bgr(bgr, det):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = det.process_image(img)
    f = r["features"]
    prof = r["radial_profile"]
    prof_ds = [float(prof[k::4].mean()) for k in range(4)]
    return {
        "hf_lf_ratio": f["hf_lf_ratio"],
        "grid_peak_prominence": f["grid_peak_prominence"],
        "axis_peak": f["axis_peak"],
        "prof0": prof_ds[0], "prof1": prof_ds[1], "prof2": prof_ds[2], "prof3": prof_ds[3],
        "mean_log_mag": float(r["spectrum"].mean()),
    }


def extract_parquet(path, label, det, limit=LIMIT):
    pf = pq.ParquetFile(path)
    rows, seen = [], 0
    for rg in range(pf.num_row_groups):
        if len(rows) >= limit:
            break
        t = pf.read_row_group(rg, columns=["image"])
        col = t.column("image")
        for i in range(len(col)):
            if len(rows) >= limit:
                break
            b = col[i]["bytes"].as_py()
            bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            try:
                f = feats_from_bgr(bgr, det)
            except Exception:
                continue
            f["label"] = label
            rows.append(f)
            seen += 1
            if seen % 500 == 0:
                print(f"  {os.path.basename(path)} label={label}: {seen}", flush=True)
    return rows


def main():
    det = SpectralDetector()
    all_rows = []
    tasks = [
        (os.path.join(DATA, "biggan_train0.parquet"), 1),
        (os.path.join(DATA, "lfw_real.parquet"), 0),
    ]
    for path, label in tasks:
        if os.path.exists(path):
            print(f"extracting {os.path.basename(path)} (label={label})")
            all_rows.extend(extract_parquet(path, label, det))
        else:
            print("missing", path)
    out = os.path.join(DATA, "features", "features.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_rows, f)
    print(f"wrote {len(all_rows)} rows to {out}")


if __name__ == "__main__":
    main()
