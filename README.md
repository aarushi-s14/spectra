# SPECTRA

Spectral deepfake detection at the speed of light. Built for the YC "Proving You're Human" RFS.

Generative models must upsample their output. Upscaling is periodic by construction, and periodicity is exactly what a Fourier transform sees. SPECTRA reads that fingerprint directly: pure NumPy, no neural network on the hot path, ~3 ms per frame on a laptop CPU.

## Results (honest splits, held-out data)

| Task | Balanced accuracy | Method |
|---|---|---|
| Video upload (CelebDF v2, 160 videos) | **0.900** | face-region spectral head, video-level |
| Live webcam (simulated, 5-frame voting) | **0.818** | single-frame face head + median voting |
| Still images (BigGAN/SD vs real photos) | 0.737 | global spectral head |
| Sora-class generated video | caught 2/3 | sharp-zone rule (see limitations) |

## How it works

1. Each frame is split into 32x32 patches, high-pass filtered (Butterworth)
2. A 2D FFT log-magnitude spectrum is computed per patch and averaged
3. Features: high/low-frequency energy ratio, radial peak prominence, axis-aligned grid-peak strength, plus center-crop face-region stats
4. Domain routing: compressed/webcam frames go to the face head, sharp photos to the global head, ultra-sharp video to a zone rule
5. Gradient-boosted heads (HistGradientBoosting) combine features into a score

Three verdicts: `likely-real` (< 0.4), `uncertain` (0.4 to 0.6), `likely-ai-generated` (> 0.6).

## Run

```sh
./run-app.sh        # serves at http://localhost:8642
```

Open http://localhost:8642, drop an image or video, or use the webcam (live scan votes over a rolling 5-frame median).

## Layout

- `spectral_core.py` — detector core: patching, FFT, features, domain routing, verdicts
- `server.py` — FastAPI backend (`/analyze/image`, `/analyze/video`), on-the-fly H.264 transcode for browser playback
- `index.html` — landing page + detector UI (single file, no build step)
- `extract_*.py` — feature extraction from datasets (BigGAN, LFW, Stable Diffusion, CelebDF, Wikipedia photos)
- `train_*.py` — head training with honest splits (odd-index videos held out)
- `evaluate.py` — batch evaluation harness
- `spectral_head*.json` — serialized trained heads (pickled sklearn inside JSON)

## Known limits

- **Sharp high-res stills**: aggregate spectral stats cannot separate modern diffusion stills from sharp real photos (all candidate features < 0.6 sigma, tested spectral slope, CFA peaks, JPEG grids, per-patch variance). Those get an honest `uncertain` verdict instead of a guess.
- **Extreme motion blur / dark scenes** wash out the spectral fingerprint (one Sora bodycam-style clip is missed).
- **Text-to-video models** (Sora/Veo class) were not in the training set; the sharp-zone rule catches high-detail generated video but is not a learned detector.
- Videos are re-encoded to H.264 in-browser-playable form on the fly (OpenCV writer; no ffmpeg dependency).
