"""Spectral deepfake detection core.

Pipeline (vectorized adaptation of the research notebooks):
  frame -> grayscale -> 256x256 -> 32x32 patches -> high-pass residual
        -> 2D FFT -> log-magnitude spectrum -> aggregate statistics -> verdict

Features:
  - radial energy profile of the spectrum (freq-band energies)
  - grid-peak prominence: periodic upscaler artifacts appear as spikes
    at rational frequencies; we measure local maxima above the radial mean
  - high/low frequency energy ratio

The verdict combines these features with a reference distribution built
from real-camera-like content (natural 1/f^2 falloff). Pure-numpy, no
neural net on this path -> fast enough for real time.
"""
from __future__ import annotations

import numpy as np

FRAME_SIZE = 256
PATCH_SIZE = 32
MAX_PATCHES = 64  # per frame; 256/32 = 64 non-overlapping patches


def _highpass(patch: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    from scipy.signal import filtfilt
    lp = patch.copy()
    lp = filtfilt(b, a, lp, axis=0)
    lp = filtfilt(b, a, lp, axis=1)
    return patch - lp


def frame_patches(gray: np.ndarray, patch_size: int = PATCH_SIZE, max_patches: int = MAX_PATCHES) -> np.ndarray:
    """Extract up to max_patches non-overlapping patches from a 2D array."""
    h, w = gray.shape
    ps = patch_size
    y0, x0 = max(0, (h - ps) // 2), max(0, (w - ps) // 2)
    ys = range(min(y0, max(0, h - ps)), max(0, h - ps) + 1, ps)
    xs = range(min(x0, max(0, w - ps)), max(0, w - ps) + 1, ps)
    patches = []
    for y in ys:
        for x in xs:
            p = gray[y:y + ps, x:x + ps]
            if p.shape == (ps, ps):
                patches.append(p)
            if len(patches) >= max_patches:
                return np.asarray(patches)
    return np.asarray(patches)


class SpectralDetector:
    def __init__(self, frame_size: int = FRAME_SIZE, patch_size: int = PATCH_SIZE,
                 cutoff: float = 2.0, order: int = 2):
        from scipy.signal import butter
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.b, self.a = butter(order, cutoff / (frame_size / 2.0), btype="high")
        ps = patch_size
        # radial frequency bin edges for a ps x ps fftshifted spectrum
        yy, xx = np.mgrid[0:ps, 0:ps]
        cy = cx = ps // 2
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        self.r = r.astype(np.int32)
        self.max_r = int(self.r.max()) + 1

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        img = img.astype(np.float32)
        h, w = img.shape
        s = min(h, w)
        img = img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        try:
            from cv2 import resize
            img = resize(img, (self.frame_size, self.frame_size))
        except ImportError:
            from PIL import Image as PILImage
            img = np.asarray(PILImage.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                             .resize((self.frame_size, self.frame_size), PILImage.LANCZOS),
                             dtype=np.float32)
        return img

    def _highpass(self, patch: np.ndarray) -> np.ndarray:
        from scipy.signal import filtfilt
        lp = patch.copy()
        lp = filtfilt(self.b, self.a, lp, axis=0)
        lp = filtfilt(self.b, self.a, lp, axis=1)
        return patch - lp

    def process_image(self, img: np.ndarray) -> dict:
        """img: HxW uint8/float BGR or RGB or grayscale array."""
        gray = self._to_gray(img)
        patches = frame_patches(gray, self.patch_size)
        if len(patches) == 0:
            raise ValueError("image too small for patching")
        spectra = np.stack([
            np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(self._highpass(p)))))
            for p in patches
        ])
        return self._features(spectra)

    def _features(self, spectra: np.ndarray) -> dict:
        # spectra: (n_patches, ps, ps) log-magnitude, fftshifted
        avg = spectra.mean(axis=0)
        ps = avg.shape[0]
        cy = cx = ps // 2

        # radial profile: mean log-magnitude per frequency radius
        profile = np.bincount(self.r.ravel(), weights=avg.ravel(), minlength=self.max_r) \
                  / np.bincount(self.r.ravel(), minlength=self.max_r)
        # high vs low frequency energy split (mid radius)
        mid = self.max_r // 2
        low_e, high_e = profile[:mid].mean(), profile[mid:].mean()
        ratio = high_e / (low_e + 1e-9)
        # grid-peak prominence: how far local maxima exceed the smoothed
        # radial baseline — periodic upsampling leaves spikes
        smooth = np.convolve(profile, np.ones(5) / 5, mode="same")
        resid = profile - smooth
        mid_band = resid[mid: mid + int(mid * 0.6)]
        peak_prom = float(mid_band.max() - mid_band.mean()) if mid_band.size else 0.0

        # axis-aligned peak strength: max on-axis magnitude minus the radial
        # baseline at the same radius — periodic upsampling grids put sharp
        # spikes exactly on the axes at rational frequencies
        axis_slopes = np.concatenate([avg[cy, cx + 1:], avg[cy, :cx],
                                      avg[cy + 1:, cx], avg[:cy, cx]])
        axis_radii = np.concatenate([np.arange(1, cx + 1)[:len(avg[cy, cx + 1:])],
                                     np.arange(cx, 0, -1),
                                     np.arange(1, cy + 1)[:len(avg[cy + 1:, cx])],
                                     np.arange(cy, 0, -1)])
        baseline_at = profile[axis_radii]
        axis_peak = float((axis_slopes - baseline_at).max())

        score = self._score(ratio, peak_prom, axis_peak)
        return {
            "score": score,
            "features": {
                "hf_lf_ratio": float(ratio),
                "grid_peak_prominence": peak_prom,
                "axis_peak": axis_peak,
            },
            "spectrum": avg,
            "radial_profile": profile,
        }

    def _score(self, ratio: float, peak_prom: float, axis_peak: float) -> float:
        """Combine features into [0,1] fake-likelihood.

        Calibrated so natural images (1/f^2 energy falloff, isotropic)
        score low; periodic upsampling grids score high. Soft logistic on
        each feature then a weighted blend.
        """
        def logistic(x, center, scale):
            return 1.0 / (1.0 + np.exp(-(x - center) / scale))
        f1 = logistic(ratio, 0.85, 0.18)          # excess high-freq energy
        f2 = logistic(peak_prom, 0.45, 0.18)      # radial spikes
        f3 = logistic(axis_peak, 2.2, 1.0)        # axis-aligned grid peaks
        s = 0.20 * f1 + 0.15 * f2 + 0.65 * f3     # axis peaks are the signal
        return float(min(1.0, max(0.0, s)))

    def verdict(self, img: np.ndarray) -> dict:
        r = self.process_image(img)
        score = r["score"]
        head = _load_head()
        frame_head = _load_frame_head()
        hf = r["features"]["hf_lf_ratio"]
        mlm = float(r["spectrum"].mean())
        # Compressed/low-res frames (webcam, video): the face-region frame head is
        # the strongest detector (0.82 balanced acc on held-out CelebDF frames).
        # High-res photos (mlm > 4.6) are a different domain — use the global head.
        if frame_head is not None and mlm <= 4.6:
            gray = self._to_gray(img)
            stats = _face_frame_stats(gray)
            if stats is not None:
                score = float(frame_head.predict_proba([stats])[0, 1])
        elif head is not None:
            prof = r["radial_profile"]
            row = [[
                r["features"]["hf_lf_ratio"],
                r["features"]["grid_peak_prominence"],
                r["features"]["axis_peak"],
                *[float(prof[k::4].mean()) for k in range(4)],
                mlm,
                0.0, 0.0, 0.0, 0.0,  # temporal std features: N/A for single frame
            ]]
            score = float(head.predict_proba(row)[0, 1])
        # Very sharp high-res camera photos (hf energy > 0.66, mean log-mag > 5.4)
        # sit above almost all training fakes; diffusion fakes rarely reach this zone
        # (<10% of SD samples). Report uncertain instead of a confident false positive.
        if hf > 0.66 and mlm > 5.4 and score > 0.5:
            score = 0.5 + (score - 0.5) * 0.15
        verdict = ("likely-ai-generated" if score > 0.6 else
                   "likely-real" if score < 0.4 else "uncertain")
        return {
            "verdict": verdict,
            "confidence": abs(score - 0.5) * 2,
            "score": score,
            "features": r["features"],
            "radial_profile": r["radial_profile"].tolist(),
            "spectrum_b64": _encode_spectrum(r["spectrum"]),
        }


_HEAD = None


def _load_head():
    global _HEAD
    if _HEAD is not None:
        return _HEAD
    import json, os, pickle
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectral_head.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    if payload["kind"] == "gb_pkl":
        _HEAD = pickle.loads(bytes.fromhex(payload["model"]))
    return _HEAD


_FRAME_HEAD = None
_FRAME_STATS = None
_DET = None


def _load_frame_head():
    global _FRAME_HEAD, _FRAME_STATS
    if _FRAME_STATS is None:
        from extract_face_video_frames import STAT_NAMES, face_frame_stats
        _FRAME_STATS = (STAT_NAMES, face_frame_stats)
    if _FRAME_HEAD is not None:
        return _FRAME_HEAD
    import json, os, pickle
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectral_head_frame_v1.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    _FRAME_HEAD = pickle.loads(bytes.fromhex(payload["model"]))
    return _FRAME_HEAD


def _face_frame_stats(gray):
    _, fn = _FRAME_STATS
    return fn(_singleton_detector(), gray)


def _singleton_detector():
    global _DET
    if _DET is None:
        _DET = SpectralDetector()
    return _DET


def _encode_spectrum(spectrum: np.ndarray) -> str:
    import base64, io
    from PIL import Image as PILImage
    norm = (spectrum - spectrum.min()) / (np.ptp(spectrum) + 1e-9)
    img8 = (norm * 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(img8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
