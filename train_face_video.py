"""Train video-level face head from per-frame face stats.

Aggregates per-frame stats to per-video mean+std (24 features), trains
HistGradientBoosting with an honest interleaved video-level split
(even-index videos train, odd-index test — no frames of a test video
are ever seen in training).
"""
import json, os, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
src = json.load(open(os.path.join(HERE, "..", "data", "features", "features_face_frames.json")))
STATS = [k for k in src[0] if k not in ("video", "vi", "label")]
assert len(STATS) == 12, STATS

by_video = {}
for r in src:
    by_video.setdefault((r["vi"], r["label"]), []).append(r)

vids = sorted(by_video.keys())
X, y, vi = [], [], []
for key in vids:
    rows = by_video[key]
    arr = np.array([[r[s] for s in STATS] for r in rows])
    X.append(np.concatenate([arr.mean(axis=0), arr.std(axis=0)]))
    y.append(rows[0]["label"])
    vi.append(key[0])
X, y, vi = np.array(X), np.array(y), np.array(vi)
print(f"{len(y)} videos ({y.sum()} syn, {len(y)-y.sum()} real), {X.shape[1]} features")

te_mask = vi % 2 == 1  # odd-index videos held out entirely
tr_mask = ~te_mask
gb = HistGradientBoostingClassifier(random_state=0)
gb.fit(X[tr_mask], y[tr_mask])

p_te = gb.predict_proba(X[te_mask])[:, 1]
acc = float(((p_te > 0.5).astype(int) == y[te_mask]).mean())
print(f"video-level test acc: {acc:.3f} on {te_mask.sum()} held-out videos")
pred = (p_te > 0.5).astype(int)
print(confusion_matrix(y[te_mask], pred))

# per-video probs for the blend evaluation
probs_all = gb.predict_proba(X)[:, 1]
out = {"kind": "gb_pkl", "model": pickle.dumps(gb).hex(), "test_acc": acc,
       "video_level": True, "n_features": int(X.shape[1]),
       "stats": STATS}
with open(os.path.join(HERE, "spectral_head_video.json"), "w") as f:
    json.dump(out, f)
np.save(os.path.join(HERE, "video_probs_debug.npy"),
        np.array([[k[0], k[1], p] for k, p in zip(vids, probs_all)]))
print("wrote spectral_head_video.json")
