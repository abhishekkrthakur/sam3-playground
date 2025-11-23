import base64
import io
import json
from pathlib import Path
from typing import Any, Dict

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from .main import load_model, run_inference
from .utils import _generate_colors, overlay_boxes, overlay_masks

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="SAM3", version="0.1.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "scripts")), name="static")


def _encode_image(image: Image.Image) -> str:
    """Convert an image to base64 for in-browser rendering."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _parse_boxes_payload(raw: str | None):
    """Parse boxes/labels JSON payload from the incoming request."""
    if not raw:
        return [], []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid boxes payload; must be JSON.") from exc

    boxes = data.get("boxes") or []
    labels = data.get("labels") or []
    if not isinstance(boxes, list):
        boxes = []
    if not isinstance(labels, list):
        labels = []

    # Pad labels with positives (1) by default.
    if len(labels) < len(boxes):
        labels = labels + [1] * (len(boxes) - len(labels))
    return boxes, labels[: len(boxes)]


def _serialize_mask_image(masks) -> str | None:
    """Merge all predicted masks into a single binary mask and encode to base64."""
    if masks is None:
        return None

    try:
        if hasattr(masks, "cpu"):
            masks = masks.cpu()
        if hasattr(masks, "numpy"):
            masks = masks.numpy()
    except Exception:
        return None

    try:
        import numpy as np  # Local import to avoid polluting global namespace.
    except Exception:
        return None

    arr = np.array(masks)
    if arr.ndim < 3 or arr.shape[0] == 0:
        return None

    merged = (arr.sum(axis=0) > 0).astype("uint8") * 255
    mask_image = Image.fromarray(merged, mode="L")
    return _encode_image(mask_image)


def _serialize_individual_masks(masks) -> list[str]:
    """Encode each predicted mask separately for client-side interaction."""
    if masks is None:
        return []

    try:
        if hasattr(masks, "cpu"):
            masks = masks.cpu()
        if hasattr(masks, "numpy"):
            masks = masks.numpy()
    except Exception:
        return []

    try:
        import numpy as np
    except Exception:
        return []

    arr = np.array(masks)
    if arr.ndim < 3 or arr.shape[0] == 0:
        return []

    encoded_masks: list[str] = []
    for single_mask in arr:
        alpha = np.clip(single_mask, 0.0, 1.0)
        alpha = (alpha * 255).astype("uint8")
        alpha_img = Image.fromarray(alpha, mode="L")
        mask_image = Image.new("RGBA", alpha_img.size, (255, 255, 255, 0))
        mask_image.putalpha(alpha_img)
        encoded_masks.append(_encode_image(mask_image))
    return encoded_masks


def _process_image(
    image: Image.Image,
    prompt: str,
    threshold: float,
    mask_threshold: float,
    boxes,
    box_labels,
) -> Dict[str, Any]:
    """Run inference and return serialized results."""
    prompt = prompt.strip() or "object"
    results = run_inference(
        image=image,
        prompt=prompt,
        threshold=threshold,
        mask_threshold=mask_threshold,
        boxes=boxes,
        box_labels=box_labels,
    )
    n_masks = int(results["masks"].shape[0])
    have_masks = n_masks > 0
    colors = _generate_colors(n_masks) if n_masks else []
    boxes = results["boxes"].tolist() if hasattr(results["boxes"], "tolist") else results["boxes"]
    mask_overlay = overlay_masks(image, results["masks"], colors) if have_masks else image
    box_overlay = overlay_boxes(image, results["masks"], boxes, colors) if have_masks else image
    mask_image = _serialize_mask_image(results["masks"]) if have_masks else None
    individual_masks = _serialize_individual_masks(results["masks"]) if have_masks else []
    return {
        "count": n_masks,
        "scores": [float(score) for score in results["scores"]],
        "boxes": [list(map(float, box)) for box in boxes],
        "colors": [list(map(int, c)) for c in colors],
        "mask_image": mask_image,
        "masks": individual_masks,
        # Keep legacy key for backward compatibility with existing UI code.
        "image": _encode_image(mask_overlay),
        "images": [
            {"id": "masks", "label": "Masks", "data": _encode_image(mask_overlay)},
            {"id": "boxes", "label": "Boxes", "data": _encode_image(box_overlay)},
        ],
    }


@app.get("/", tags=["ui"])
async def home(request: Request):
    """Render the playground UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.on_event("startup")
async def warm_models():
    """Load SAM3 model into memory on startup; fail fast if unavailable."""
    await run_in_threadpool(load_model)


@app.post("/process", tags=["inference"])
async def process_image(
    image: UploadFile = File(...),
    prompt: str = Form(None),
    threshold: float = Form(0.5),
    mask_threshold: float = Form(0.5),
    boxes: str | None = Form(None),
):
    """Handle image uploads and return segmentation overlays."""
    try:
        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Upload is empty.")
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unsupported image format.") from exc

    boxes_list, box_labels = _parse_boxes_payload(boxes)
    payload = await run_in_threadpool(
        _process_image, pil_image, prompt, threshold, mask_threshold, boxes_list, box_labels
    )
    return JSONResponse({"status": "ok", **payload})


@app.post("/process-video", tags=["inference"])
async def process_video(
    video: UploadFile = File(...),
    prompt: str = Form(None),
    threshold: float = Form(0.5),
    mask_threshold: float = Form(0.5),
):
    """Handle video uploads and return segmentation masks for each frame."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames in video: {total_frames}, fps: {fps}")

    frame_results = []
    frame_idx = 0
    success, frame = cap.read()
    while success and frame_idx < 10:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        payload = await run_in_threadpool(
            _process_image, pil_image, prompt or "object", threshold, mask_threshold, [], []
        )
        frame_results.append({
            "frame": frame_idx,
            "mask_image": payload["mask_image"],
            "image": payload["image"],
        })
        frame_idx += 1
        success, frame = cap.read()
    cap.release()
    
    print(f"Finished processing {frame_idx} frames for video prompt '{prompt}'")

    return JSONResponse({
        "status": "ok",
        "frames": frame_results,
        "total_frames": total_frames,
        "fps": fps
    })


@app.post("/video-metadata", tags=["inference"])
async def video_metadata(
    video: UploadFile = File(...)
):
    import cv2
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps else 0
    cap.release()
    return {"total_frames": total_frames, "fps": fps, "duration": duration}


@app.get("/healthz", tags=["meta"])
async def healthcheck():
    """Simple health endpoint for uptime checks."""
    return {"status": "ok"}
