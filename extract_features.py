"""
Train the spectral classifier on labeled data (DeepFakeFace: wiki=real, insight=fake).

Approach: for each image, extract the 3 spectral features (hf_lf_ratio,
grid_peak_prominence, axis_peak) plus richer extras (full radial profile
downsampled), then fit logistic regression + gradient boosting. This keeps
inference at the same ~2ms cost — no neural net needed.
"""
import sys, os, glob, json, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_core import SpectralDetector, frame_patches

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
EXTRACT_DIR = os.path.join(DATA_DIR, "features")


def extract_from_folder(folder, label, limit_per_dir=4000, seed=0):
    rng = np.random.default_rng(seed)
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if os.path.splitext(n)[1].lower() in {".jpg", ".jpeg", ".png"}:
                files.append(os.path.join(root, n))
    rng.shuffle(files)
    files = files[:limit_per_dir]
    det = SpectralDetector()
    rows = []
    t0 = time.perf_counter()
    for i, p in enumerate(files):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            r = det.process_image(img)
        except Exception:
            continue
        f = r["features"]
        prof = r["radial_profile"]
        prof_ds = np.array([prof[k::4].mean() for k in range(4)])  # 4-band summary
        rows.append({
            "label": label,
            "hf_lf_ratio": f["hf_lf_ratio"],
            "grid_peak_prominence": f["grid_peak_prominence"],
            "axis_peak": f["axis_peak"],
            "prof0": float(prof_ds[0]), "prof1": float(prof_ds[1]),
            "prof2": float(prof_ds[2]), "prof3": float(prof_ds[3]),
            "mean_log_mag": float(r["spectrum"].mean()),
        })
        if i % 500 == 0 and i:
            print(f"  {folder.split('/')[-1]}: {i}/{len(files)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return rows


def main():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    all_rows = []
    for name, label in [("wiki", 0), ("insight", 1)]:
        folder = os.path.join(DATA_DIR, name)
        if not os.path.isdir(folder):
            print(f"MISSING {folder} — extract the zip first")
            continue
        print(f"extracting features from {name}/ ...")
        rows = extract_from_folder(folder, label)
        print(f"  got {len(rows)} rows")
        all_rows.extend(rows)
    if not all_rows:
        sys.exit(1)
    out = os.path.join(EXTRACT_DIR, "features.json")
    with open(out, "w") as f:
        json.dump(all_rows, f)
    print(f"wrote {len(all_rows)} feature rows to {out}")


if __name__ == "__main__":
    main()
