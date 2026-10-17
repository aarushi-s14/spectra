"""Train the combined spectral head on merged features.

Sources: BigGAN fakes + LFW reals + SD fakes + CelebDF real video frames.
Video rows (12-col schema with temporal stds) are padded with 0 stds like image rows.
Downsamples fakes to match real count to avoid class bias.
"""
import json, os, pickle, glob
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "..", "data", "features")

rows = []
img_fakes = json.load(open(os.path.join(FEAT, "features.json")))          # biggan=1
img_reals = [r for r in img_fakes if r["label"] == 0]
img_fakes = [r for r in img_fakes if r["label"] == 1]
sd_fakes = json.load(open(os.path.join(FEAT, "features_sd.json")))        # label=1
vid_reals = json.load(open(os.path.join(FEAT, "features_realvideo.json")))  # label=0
wiki_reals = json.load(open(os.path.join(FEAT, "features_wiki.json")))    # label=0

rows += img_reals + img_fakes + sd_fakes + vid_reals + wiki_reals
n_real, n_fake = sum(1 for r in rows if r["label"] == 0), sum(1 for r in rows if r["label"] == 1)
print(f"before balance: real={n_real} fake={n_fake}")

# balance classes by downsampling fakes
rng = np.random.RandomState(0)
if n_fake > n_real:
    fake_idx = np.where([r["label"] == 1 for r in rows])[0]
    drop = set(rng.choice(fake_idx, n_fake - n_real, replace=False))
    rows = [r for i, r in enumerate(rows) if i not in drop]
n_real, n_fake = sum(1 for r in rows if r["label"] == 0), sum(1 for r in rows if r["label"] == 1)
print(f"after balance: real={n_real} fake={n_fake}")

F = ["hf_lf_ratio", "grid_peak_prominence", "axis_peak",
     "prof0", "prof1", "prof2", "prof3", "mean_log_mag",
     "hf_lf_ratio_std", "grid_peak_prominence_std", "axis_peak_std", "mean_log_mag_std"]
X = np.array([[r[k] if k in r else 0.0 for k in F] for r in rows])
y = np.array([r["label"] for r in rows])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
gb = HistGradientBoostingClassifier(random_state=0)
gb.fit(Xtr, ytr)
acc = gb.score(Xte, yte)
p = gb.predict_proba(Xte)[:, 1]
print(f"combined head test acc: {acc:.4f}")
print(f"real FPR: {float((p[yte==0]>0.5).mean()):.3f}  fake TPR: {float((p[yte==1]>0.5).mean()):.3f}")
print(confusion_matrix(yte, gb.predict(Xte)))

payload = {"kind": "gb_pkl", "model": pickle.dumps(gb).hex(), "features": F, "test_acc": float(acc)}
with open(os.path.join(HERE, "spectral_head.json"), "w") as f:
    json.dump(payload, f)
print("wrote spectral_head.json")
