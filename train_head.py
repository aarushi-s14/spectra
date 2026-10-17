"""Fit the classifier on extracted features and write spectral_head.json."""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "..", "data", "features", "features.json")
OUT = os.path.join(HERE, "spectral_head.json")

FEATURES = ["hf_lf_ratio", "grid_peak_prominence", "axis_peak",
            "prof0", "prof1", "prof2", "prof3", "mean_log_mag"]


def main():
    with open(FEAT) as f:
        rows = json.load(f)
    X = np.array([[r[k] for k in FEATURES] for r in rows])
    y = np.array([r["label"] for r in rows])
    print(f"dataset: {len(y)} rows, {y.sum()} fake / {len(y)-y.sum()} real")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr, ytr)
    acc_lr = lr.score(Xte, yte)
    print(f"logistic regression test acc: {acc_lr:.4f}")

    gb = HistGradientBoostingClassifier(random_state=0)
    gb.fit(Xtr, ytr)
    acc_gb = gb.score(Xte, yte)
    print(f"gradient boosting test acc:  {acc_gb:.4f}")

    best = gb if acc_gb >= acc_lr else lr
    kind = "gb" if best is gb else "lr"
    # serialize as plain coefficients (lr) or node-threshold dumps (gb)
    if kind == "lr":
        payload = {"kind": "lr", "coef": lr.coef_[0].tolist(),
                   "intercept": float(lr.intercept_[0]), "features": FEATURES,
                   "test_acc": float(acc_lr)}
    else:
        # sklearn boosters are picky to serialize portably; use their internal arrays
        import pickle
        payload = {"kind": "gb_pkl", "model": pickle.dumps(gb).hex(),
                   "features": FEATURES, "test_acc": float(acc_gb)}
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"wrote {OUT} (kind={kind}, test_acc={payload['test_acc']:.4f})")


if __name__ == "__main__":
    main()
