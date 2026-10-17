"""FastAPI backend for spectral deepfake detection."""
from __future__ import annotations

import base64
import io
import json
import pickle
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from spectral_core import SpectralDetector, _load_head as spectral_head, frame_patches
from extract_face_video_frames import STAT_NAMES, face_frame_stats


def spectral_face_head():
    path = Path(__file__).parent / "spectral_head_video.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return pickle.loads(bytes.fromhex(payload["model"]))


def face_video_prob(detector, tmp_path, head):
    """Video-level face-region probability, computed identically to training:
    12 evenly spaced frames, center crop stats, mean+std over frames (24 feats)."""
    cap = cv2.VideoCapture(str(tmp_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, max(total - 1, 0), min(12, total)).astype(int)
    per_frame = []
    for w in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(w))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = detector._to_gray(rgb)
        stats = face_frame_stats(detector, gray)
        if stats is None:
            continue
        per_frame.append(stats)
    cap.release()
    if not per_frame:
        return None
    arr = np.array(per_frame)
    row = np.concatenate([arr.mean(axis=0), arr.std(axis=0)])[None, :]
    return float(head.predict_proba(row)[0, 1])


app = FastAPI(title="Spectral Deepfake Detector")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

detector = SpectralDetector()

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((Path(__file__).parent / "index.html").read_text())


@app.get("/playback")
def playback() -> HTMLResponse:
    return HTMLResponse((Path(__file__).parent / "playback.html").read_text())


@app.get("/fakes/{name}")
def fake_video(name: str) -> FileResponse:
    """Serve demo deepfake videos for the phone-mirror playback trick."""
    if not (name.endswith(".mp4") and "/" not in name and ".." not in name):
        raise HTTPException(status_code=404)
    for sub in ("syn", "real"):
        p = Path(__file__).parent.parent / "data" / "celebdf" / sub / name
        if p.exists():
            return FileResponse(p, media_type="video/mp4")
    raise HTTPException(status_code=404)


@app.get("/media/{name}")
def media_file(name: str) -> FileResponse:
    """Serve previously uploaded videos back to the player."""
    if "/" in name or ".." in name:
        raise HTTPException(status_code=404)
    p = Path(__file__).parent / "media" / name
    if not p.is_file():
        raise HTTPException(status_code=404)
    media_type = "video/webm" if p.suffix == ".webm" else "video/mp4"
    return FileResponse(p, media_type=media_type)


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not decode image"}, status_code=400)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    result = detector.verdict(rgb)
    elapsed = time.perf_counter() - t0
    return {**result, "latency_ms": round(elapsed * 1000, 1),
            "filename": file.filename}


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), max_frames: int = 30):
    data = await file.read()
    media_dir = Path(__file__).parent / "media"
    media_dir.mkdir(exist_ok=True)
    ext = Path(file.filename or "").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, dir=media_dir, delete=False, prefix="up_")
    tmp.write(data)
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    scores, timeline, spectrum_acc = [], [], None
    frame_feats = []
    n = 0
    t0 = time.perf_counter()
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = detector.process_image(rgb)
        scores.append(r["score"])
        timeline.append(r["features"])
        prof = r["radial_profile"]
        frame_feats.append([
            r["features"]["hf_lf_ratio"], r["features"]["grid_peak_prominence"],
            r["features"]["axis_peak"], *[float(prof[k::4].mean()) for k in range(4)],
            float(r["spectrum"].mean()),
        ])
        spec = r["spectrum"]
        spectrum_acc = spec if spectrum_acc is None else spectrum_acc + spec
        n += 1
    cap.release()
    if not scores:
        return JSONResponse({"error": "no frames decoded"}, status_code=400)
    elapsed = time.perf_counter() - t0
    spectrum_acc /= len(scores)

    score = float(np.mean(scores))
    head = spectral_head()
    if head is not None:
        ff = np.array(frame_feats)
        stds = ff.std(axis=0)
        rows = np.hstack([ff, stds[None, :].repeat(ff.shape[0], axis=0)])[:, :12]
        score = 0.5 * score + 0.5 * float(np.mean(head.predict_proba(rows)[:, 1]))
    # Sharp-zone rule: no known real video (0% of 960 compressed-real frames, max
    # mlm 5.23 / hf 0.60) reaches hf>0.63 & mlm>5.3, but Sora-class generated
    # video does. The face head is out-of-domain there (trained on compressed
    # webcam faces), so decide from the zone before it can override.
    ff_arr = np.array(frame_feats)
    v_hf, v_mlm = float(ff_arr[:, 0].mean()), float(ff_arr[:, 7].mean())
    sharp_zone = v_hf > 0.63 and v_mlm > 5.3
    if sharp_zone:
        score, conf = 0.85, 0.6
        verdict = "likely-ai-generated"
    else:
        face_head = spectral_face_head()
        if face_head is not None:
            face_p = face_video_prob(detector, tmp.name, face_head)
            if face_p is not None:
                # face-region head wins empirically (0.90 balanced acc on CelebDF
                # held-out videos vs ~0.81 for any blend with the global head)
                score = face_p
        conf = abs(score - 0.5) * 2
        verdict = ("likely-ai-generated" if score > 0.6 else
                   "likely-real" if score < 0.4 else "uncertain")
    # Chrome's <video> only plays H.264/VP9/AV1 — re-encode if the upload
    # uses another codec (e.g. MPEG-4 Part 2, common in CelebDF-style mp4s).
    out_name = Path(tmp.name).stem + "_h264.mp4"
    out_path = media_dir / out_name
    cap = cv2.VideoCapture(tmp.name)
    vw = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if vw is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps,
                                 (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    cap.release()
    if vw is not None:
        vw.release()
    media_url = "/media/" + out_name if out_path.exists() else "/media/" + Path(tmp.name).name
    return {
        "verdict": verdict,
        "confidence": conf,
        "score": score,
        "frames_analyzed": len(scores),
        "latency_ms": round(elapsed * 1000, 1),
        "ms_per_frame": round(elapsed * 1000 / len(scores), 2),
        "fps": round(len(scores) / elapsed, 1),
        "timeline_scores": [round(float(s), 4) for s in scores],
        "spectrum_b64": _encode(spectrum_acc),
        "media_url": media_url,
        "filename": file.filename,
    }


def _encode(spectrum: np.ndarray) -> str:
    norm = (spectrum - spectrum.min()) / (np.ptp(spectrum) + 1e-9)
    img8 = (norm * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO))
    return base64.b64encode(buf.tobytes()).decode()
