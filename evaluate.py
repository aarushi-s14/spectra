"""
Evaluation harness: run the spectral detector over real data and report
separation. This is the tool we use to calibrate thresholds properly.

Data: point DATA_ROOT at folders with subdirs 'real' and a fake class
(e.g. 'veo3'). Images or videos both supported.
"""
import sys, glob, os, time
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_core import SpectralDetector


def iter_images(folder, limit=None):
    exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in sorted(glob.glob(os.path.join(folder, "*"))) if os.path.splitext(f)[1].lower() in exts]
    return files[:limit] if limit else files


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/Users/aarushishukla/Desktop/yc-hackathon/Detecting-Deepfakes-src"
    d = SpectralDetector()
    results = {}
    for cls_dir in sorted(glob.glob(os.path.join(root, "*"))):
        cls = os.path.basename(cls_dir)
        if not os.path.isdir(cls_dir) or cls.startswith("sample") or cls.startswith("miss"):
            continue
        scores, feats = [], []
        t0 = time.perf_counter()
        for p in iter_images(cls_dir, limit=200):
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            r = d.process_image(img)
            scores.append(r["score"])
            feats.append(r["features"])
        if not scores:
            continue
        dt = time.perf_counter() - t0
        results[cls] = dict(
            n=len(scores), mean=float(np.mean(scores)), std=float(np.std(scores)),
            axis_peak=float(np.mean([f["axis_peak"] for f in feats])),
            hf_lf=float(np.mean([f["hf_lf_ratio"] for f in feats])),
            ms_per_img=dt * 1000 / len(scores),
        )
    print(f"{'class':<20}{'n':>5}{'mean':>8}{'std':>8}{'axis_pk':>9}{'hf/lf':>8}{'ms/img':>8}")
    for cls, r in results.items():
        print(f"{cls:<20}{r['n']:>5}{r['mean']:>8.3f}{r['std']:>8.3f}{r['axis_peak']:>9.2f}{r['hf_lf']:>8.3f}{r['ms_per_img']:>8.1f}")

    fakes = [r["mean"] for c, r in results.items() if c.lower() not in ("real",) and "real" not in c.lower()]
    reals = [r["mean"] for c, r in results.items() if "real" in c.lower()]
    if fakes and reals:
        print(f"\nfake-class mean: {np.mean(fakes):.3f}   real-class mean: {np.mean(reals):.3f}")
        print(f"separation: {np.mean(fakes) - np.mean(reals):+.3f}")


if __name__ == "__main__":
    main()
