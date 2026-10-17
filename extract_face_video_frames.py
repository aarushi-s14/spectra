"""Extract per-frame face-region spectral stats from CelebDF videos.

Center 128x128 crop of the 256x256 frame (face region), 16 patches,
12 spectral stats per frame. Saves one row per frame with a video index
so training can aggregate or split at video level honestly.
"""
import sys, os, json, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_core import SpectralDetector, frame_patches

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FRAMES_PER_VIDEO = 12
VIDEOS_PER_CLASS = 80

STAT_NAMES = [
    "spec_max", "spec_std", "low_mean", "high_mean",
    "prof4", "prof8", "prof12", "prof16",
    "axis_e8", "axis_w8", "axis_s8", "axis_n8",
]


def face_frame_stats(det, gray):
    center = gray[64:192, 64:192]
    patches = frame_patches(center, 32)[:16]
    if len(patches) == 0:
        return None
    specs = np.stack([
        np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(det._highpass(p)))))
        for p in patches
    ])
    avg = specs.mean(axis=0)
    c = 16
    yy, xx = np.mgrid[0:32, 0:32]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2).astype(int)
    prof = (np.bincount(r.ravel(), weights=avg.ravel(), minlength=20)
            / np.bincount(r.ravel(), minlength=20))
    return [
        float(avg.max()), float(specs.std()),
        float(specs[:, :8].mean()), float(specs[:, -8:].mean()),
        float(prof[4]), float(prof[8]), float(prof[12]), float(prof[16]),
        float(avg[c, c + 8] - prof[8]), float(avg[c, c - 8] - prof[8]),
        float(avg[c + 8, c] - prof[8]), float(avg[c - 8, c] - prof[8]),
    ]


def main():
    det = SpectralDetector()
    rows = []
    for sub, label in [("celebdf/real", 0), ("celebdf/syn", 1)]:
        vids = sorted(glob.glob(os.path.join(DATA, sub, "*.mp4")))[:VIDEOS_PER_CLASS]
        print(f"{sub}: {len(vids)} videos", flush=True)
        for vi, v in enumerate(vids):
            vid = os.path.basename(v)
            cap = cv2.VideoCapture(v)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                continue
            idxs = np.linspace(0, max(total - 1, 0), min(FRAMES_PER_VIDEO, total)).astype(int)
            for w in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(w))
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = det._to_gray(rgb)
                stats = face_frame_stats(det, gray)
                if stats is None:
                    continue
                rows.append({"video": vid, "vi": vi, "label": label,
                             **{n: s for n, s in zip(STAT_NAMES, stats)}})
            cap.release()
            if (vi + 1) % 20 == 0:
                print(f"  {vi+1}/{len(vids)}", flush=True)
    out = os.path.join(DATA, "features", "features_face_frames.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} frame rows to {out}")


if __name__ == "__main__":
    main()
