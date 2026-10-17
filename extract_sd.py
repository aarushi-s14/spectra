"""Extract spectral features from SD(flickr30k-SD parquet) = modern diffusion fakes.
Merges with existing BigGAN+LFW features for a combined image head."""
import sys, os, json
import numpy as np
import cv2
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_core import SpectralDetector
from extract_parquet import feats_from_bgr, DATA, LIMIT

def main():
    det = SpectralDetector()
    path = os.path.join(DATA, "sd_train0.parquet")
    pf = pq.ParquetFile(path)
    rows = []
    for rg in range(pf.num_row_groups):
        if len(rows) >= LIMIT:
            break
        t = pf.read_row_group(rg, columns=["image"])
        col = t.column("image")
        for i in range(len(col)):
            if len(rows) >= LIMIT:
                break
            b = col[i]["bytes"].as_py()
            bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            try:
                f = feats_from_bgr(bgr, det)
            except Exception:
                continue
            f["label"] = 1
            rows.append(f)
            if len(rows) % 500 == 0:
                print(f"  sd: {len(rows)}", flush=True)
    out = os.path.join(DATA, "features", "features_sd.json")
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} rows to {out}")

if __name__ == "__main__":
    main()
