"""
Extract spectral features from video files (CelebDF v2: real vs synthesis).
Samples N frames per video, computes features per frame, aggregates to
per-video stats (mean + std of each feature across frames).
"""
import sys, os, json, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_core import SpectralDetector

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FRAMES_PER_VIDEO = 12
VIDEOS_PER_CLASS = 80

BASE_FEATURES = ["hf_lf_ratio", "grid_peak_prominence", "axis_peak",
                 "prof0", "prof1", "prof2", "prof3", "mean_log_mag"]


def video_rows(path, label, det):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, max(total - 1, 0), min(FRAMES_PER_VIDEO, total)).astype(int)
    frame_scores = []
    for want in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(want))
        ok, frame = cap.read()
        if not ok:
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            r = det.process_image(img)
        except Exception:
            continue
        f = r["features"]
        prof = r["radial_profile"]
        frame_scores.append([
            f["hf_lf_ratio"], f["grid_peak_prominence"], f["axis_peak"],
            *[float(prof[k::4].mean()) for k in range(4)],
            float(r["spectrum"].mean()),
        ])
    cap.release()
    if not frame_scores:
        return []
    fs = np.array(frame_scores)
    row = {"label": label, "n_frames": len(fs)}
    for i, name in enumerate(BASE_FEATURES):
        row[name] = float(fs[:, i].mean())
        row[name + "_std"] = float(fs[:, i].std())
    return [row]


def main():
    det = SpectralDetector()
    rows = []
    for sub, label in [("celebdf/real", 0), ("celebdf/syn", 1)]:
        vids = sorted(glob.glob(os.path.join(DATA, sub, "*.mp4")))[:VIDEOS_PER_CLASS]
        print(f"{sub}: {len(vids)} videos", flush=True)
        for i, v in enumerate(vids):
            rows.extend(video_rows(v, label, det))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(vids)}", flush=True)
    out = os.path.join(DATA, "features", "features_video.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
