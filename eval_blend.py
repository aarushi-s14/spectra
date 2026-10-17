"""Evaluate blend weights between global head and face video head over all 160 CelebDF videos.

Global score: combined 12-col head on 30-frame full-frame features (as in server).
Face score: spectral_head_video.json on 12-frame center-crop mean+std.
Prints balanced accuracy at several face blend weights.
"""
import sys, os, json, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_core import SpectralDetector, _load_head
from extract_face_video_frames import face_frame_stats
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FPH = 12

det = SpectralDetector()
ghead = _load_head()
fhead = pickle.loads(bytes.fromhex(
    json.load(open(os.path.join(HERE, "spectral_head_video.json")))["model"]))

def global_probs(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), min(30, total)).astype(int)
    feats = []
    for w in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(w))
        ok, frame = cap.read()
        if not ok:
            continue
        r = det.process_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prof = r["radial_profile"]
        feats.append([r["features"]["hf_lf_ratio"], r["features"]["grid_peak_prominence"],
                      r["features"]["axis_peak"], *[float(prof[k::4].mean()) for k in range(4)],
                      float(r["spectrum"].mean())])
    cap.release()
    if not feats:
        return None
    ff = np.array(feats)
    stds = ff.std(axis=0)
    rows = np.hstack([ff, stds[None, :].repeat(ff.shape[0], axis=0)])[:, :12]
    return float(np.mean(ghead.predict_proba(rows)[:, 1]))

def face_prob(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), min(FPH, total)).astype(int)
    rows = []
    for w in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(w))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = det._to_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        s = face_frame_stats(det, gray)
        if s is not None:
            rows.append(s)
    cap.release()
    if not rows:
        return None
    arr = np.array(rows)
    return float(fhead.predict_proba(
        np.concatenate([arr.mean(axis=0), arr.std(axis=0)])[None, :])[0, 1])

rows = []
for sub, label in [("real", 0), ("syn", 1)]:
    vids = sorted(glob.glob(os.path.join(DATA, "celebdf", sub, "*.mp4")))[:80]
    for i, v in enumerate(vids):
        gp, fp = global_probs(v), face_prob(v)
        if gp is None or fp is None:
            continue
        rows.append((label, i, gp, fp))
    print(f"{sub} done", flush=True)

np.save(os.path.join(HERE, "blend_eval.npy"), np.array(rows))
for wf in [0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]:
    blended = [(1 - wf) * gp + wf * fp for _, _, gp, fp in rows]
    y = np.array([r[0] for r in rows])
    p = np.array(blended)
    tnr = float(((p[y == 0] <= 0.5)).mean())
    tpr = float(((p[y == 1] > 0.5)).mean())
    print(f"face_w={wf}: balanced_acc={(tnr + tpr) / 2:.3f}  TNR={tnr:.3f} TPR={tpr:.3f}")
