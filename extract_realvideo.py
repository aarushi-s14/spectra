"""Extract per-frame spectral features from CelebDF REAL videos.

These are H.264-compressed video frames — same domain as webcam uploads —
used to teach the image head what compressed real footage looks like.
"""
import sys, os, json, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_core import SpectralDetector
from extract_parquet import feats_from_bgr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FRAMES_PER_VIDEO = 12
VIDEOS = 80

def main():
    det = SpectralDetector()
    rows = []
    vids = sorted(glob.glob(os.path.join(DATA, "celebdf", "real", "*.mp4")))[:VIDEOS]
    print(f"{len(vids)} real videos", flush=True)
    for i, v in enumerate(vids):
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release(); continue
        idxs = np.linspace(0, max(total - 1, 0), min(FRAMES_PER_VIDEO, total)).astype(int)
        for w in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(w))
            ok, frame = cap.read()
            if not ok:
                continue
            try:
                f = feats_from_bgr(frame, det)
            except Exception:
                continue
            f["label"] = 0
            rows.append(f)
        cap.release()
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(vids)} ({len(rows)} frames)", flush=True)
    out = os.path.join(DATA, "features", "features_realvideo.json")
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} rows to {out}")

if __name__ == "__main__":
    main()
