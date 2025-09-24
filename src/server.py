import os
import warnings
import subprocess
import json
from dataclasses import asdict
from time import perf_counter

import torch
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.requests import Request
from starlette.responses import Response

from util import detect_measures, generate_random_id, is_true

# Suppress noisy deprecation warning emitted by YOLOv5 about pkg_resources
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
# Suppress YOLOv5 autocast deprecation warning on newer PyTorch
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*deprecated.*",
    category=FutureWarning,
)

app = FastAPI(
    title="Measure Detector",
    description="YOLOv5-based Measure Detector Microservice",
    version="1.2.0",
    contact={"name": "Simon Waloschek", "email": "waloschek@pm.me"},
)

# CORS for browser clients; tighten origins as needed in deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def get_health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": app.version})


@app.post("/json")
async def post_json(
    files: List[UploadFile] = File(...),
    expand: Optional[str] = Form(None),
    trim: Optional[str] = Form(None),
    auto: Optional[str] = Form(None),
    pretty: Optional[str] = Form(None),
) -> Response:
    start = perf_counter()
    expand = is_true(expand)
    trim = is_true(trim)
    auto = is_true(auto)
    pretty = is_true(pretty)

    results = []
    for uf in files:
        try:
            measures, page_type, type_conf = detect_measures(
                app.state.model, uf, expand=expand, trim=trim, auto=auto
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        results.append(
            {
                "filename": getattr(uf, "filename", None),
                "type": page_type,
                "type_confidence": round(type_conf, 3),
                "measures": [asdict(measure) for measure in measures],
            }
        )

    process_time = round((perf_counter() - start) * 1000)
    payload = {"process_time": process_time, "results": results}
    if pretty:
        return Response(
            content=json.dumps(payload, indent=2, ensure_ascii=False),
            media_type="application/json",
        )
    return JSONResponse(payload)


@app.post("/mei")
async def post_mei(
    files: List[UploadFile] = File(...),
    expand: Optional[str] = Form(None),
    trim: Optional[str] = Form(None),
    auto: Optional[str] = Form(None),
    pretty: Optional[str] = Form(None),
) -> Response:
    start = perf_counter()
    expand = is_true(expand)
    trim = is_true(trim)
    auto = is_true(auto)
    pretty = is_true(pretty)

    # Build MEI XML
    import xml.etree.ElementTree as ET

    mei = ET.Element("mei", attrib={"xmlns": "http://www.music-encoding.org/ns/mei"})
    music = ET.SubElement(mei, "music")
    body = ET.SubElement(music, "body")
    mdiv = ET.SubElement(body, "mdiv")
    score = ET.SubElement(mdiv, "score")
    section = ET.SubElement(score, "section")

    facsimile = ET.SubElement(mei, "facsimile")
    page_counter = 0
    measure_global_idx = 1
    import numpy as _np  # local import to avoid polluting namespace
    import cv2 as _cv2
    import re as _re

    def _natural_key(name: str):
        parts = _re.split(r"(\d+)", name)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    # sort files alphanumerically by filename
    files_sorted = sorted(
        files, key=lambda uf: _natural_key(getattr(uf, "filename", ""))
    )

    for uf in files_sorted:
        # Read bytes once to get image size, then reset for detector
        data = uf.file.read()
        img = _cv2.imdecode(_np.frombuffer(data, dtype=_np.uint8), _cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data uploaded")
        height, width = img.shape[:2]
        uf.file.seek(0)

        measures, page_type, type_conf = detect_measures(
            app.state.model, uf, expand=expand, trim=trim, auto=auto
        )

        page_counter += 1
        surface = ET.SubElement(
            facsimile,
            "surface",
            attrib={
                "xml:id": f"surface_{page_counter}",
                "n": str(page_counter),
                "ulx": "0",
                "uly": "0",
                "lrx": str(width - 1),
                "lry": str(height - 1),
            },
        )
        ET.SubElement(
            surface,
            "graphic",
            attrib={
                "xml:id": f"graphic_{page_counter}",
                "target": getattr(uf, "filename", "image"),
                "width": f"{width}px",
                "height": f"{height}px",
            },
        )

        # Add zones and measures for this page
        for m in measures:
            x1 = int(round(m.bbox.x1 * width))
            y1 = int(round(m.bbox.y1 * height))
            x2 = int(round(m.bbox.x2 * width))
            y2 = int(round(m.bbox.y2 * height))

            zone_id = f"zone_{measure_global_idx}"
            ET.SubElement(
                surface,
                "zone",
                attrib={
                    "xml:id": zone_id,
                    "type": "measure",
                    "ulx": str(x1),
                    "uly": str(y1),
                    "lrx": str(x2),
                    "lry": str(y2),
                },
            )

            ET.SubElement(
                section,
                "measure",
                attrib={
                    "xml:id": f"measure_{measure_global_idx}",
                    "n": str(measure_global_idx),
                    "label": str(measure_global_idx),
                    "facs": f"#{zone_id}",
                },
            )
            measure_global_idx += 1

    # Serialize XML
    if pretty:
        try:
            ET.indent(mei, space="  ")  # Python >=3.9
        except Exception:
            pass
    xml_bytes = ET.tostring(mei, encoding="utf-8")
    process_time = round((perf_counter() - start) * 1000)
    headers = {"X-Process-Time": str(process_time)}
    return Response(content=xml_bytes, media_type="application/xml", headers=headers)


@app.post("/debug")
async def post_debug(
    file: UploadFile = File(...),
    expand: Optional[str] = Form(None),
    trim: Optional[str] = Form(None),
    auto: Optional[str] = Form(None),
) -> Response:
    start = perf_counter()
    expand = is_true(expand)
    trim = is_true(trim)
    auto = is_true(auto)

    try:
        img, page_type, type_conf = detect_measures(
            app.state.model, file, expand=expand, trim=trim, auto=auto, debug=True
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    headers = {"X-Process-Time": str(round((perf_counter() - start) * 1000))}
    return Response(img, media_type="image/jpeg", headers=headers)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = perf_counter()
    pid = os.getpid()

    rid = generate_random_id()
    logger.info(
        f'rid={rid} ({request.client.host}) requested endpoint "{request.url.path}" on worker with pid [{pid}]"'
    )
    response = await call_next(request)
    process_time = round((perf_counter() - start_time) * 1000)
    logger.info(
        f'rid={rid} completed in {process_time}ms, status code "{response.status_code}"'
    )

    return response


@app.on_event("startup")
async def load_model():
    pid = os.getpid()
    repo_id = os.getenv("HF_REPO_ID", "sonovice/measure-detector")
    model_filename = os.getenv("HF_MODEL_FILENAME", "model.pt")
    revision = os.getenv("HF_REVISION", "main")

    # Prefer pre-baked weights inside the image; fallback to HF if missing (e.g., local dev)
    preferred_path = os.getenv("MODEL_PATH", "/app/model.pt")
    if os.path.exists(preferred_path):
        weights_path = preferred_path
        logger.info(f"Using pre-baked model weights at {weights_path}")
    else:
        logger.info(
            f'Downloading model weights from Hugging Face: repo_id="{repo_id}", filename="{model_filename}", revision="{revision}"'
        )
        weights_path = hf_hub_download(
            repo_id=repo_id, filename=model_filename, revision=revision
        )

    # Prefer local YOLOv5 if baked into image; otherwise fall back to GitHub
    yolo_repo = "/app/yolov5" if os.path.exists("/app/yolov5") else "ultralytics/yolov5"
    yolo_source = "local" if yolo_repo.startswith("/") else "github"

    # Try devices in order: CUDA, MPS (macOS), then CPU. Fall back automatically on errors.
    devices_to_try = []
    try:
        if torch.cuda.is_available():
            devices_to_try.append("cuda")
    except Exception:
        pass
    try:
        # Prefer CPU over unstable MPS on macOS unless explicitly overridden
        if os.getenv("FORCE_DEVICE", "").lower() == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                devices_to_try.append("mps")
    except Exception:
        pass
    devices_to_try.append("cpu")

    model = None
    last_error = None
    for device in devices_to_try:
        try:
            if device == "cpu":
                # get real number of CPU cores from docker container using nproc
                result = subprocess.run(["nproc"], stdout=subprocess.PIPE)
                num_cpus = int(result.stdout)
                workers_env = os.getenv("WORKERS")
                try:
                    num_workers = int(workers_env) if workers_env is not None else 1
                except (TypeError, ValueError):
                    num_workers = 1
                if num_workers < 1:
                    num_workers = 1
                num_cpus_per_worker = max(1, num_cpus // num_workers)
                torch.set_num_threads(num_cpus_per_worker)
                logger.info(
                    f"Loading model for pid [{pid}] on device [{device}] using {num_cpus_per_worker} CPU cores..."
                )
            else:
                logger.info(f"Loading model for pid [{pid}] on device [{device}]...")

            # Allowlist legacy numpy reconstruct for PyTorch >=2.6 safe loading
            try:
                import numpy as _np  # type: ignore
                from torch.serialization import add_safe_globals as _add_safe_globals  # type: ignore

                _add_safe_globals([_np.core.multiarray._reconstruct])
            except Exception:
                pass

            _orig_torch_load = torch.load
            try:

                def _torch_load_with_weights(*args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    return _orig_torch_load(*args, **kwargs)

                torch.load = _torch_load_with_weights  # type: ignore[assignment]

                if yolo_source == "local":
                    model = torch.hub.load(
                        yolo_repo,
                        "custom",
                        weights_path,
                        source="local",
                        device=device,
                        _verbose=False,
                    )
                else:
                    model = torch.hub.load(
                        yolo_repo,
                        "custom",
                        path=weights_path,
                        source="github",
                        device=device,
                        trust_repo=True,
                        _verbose=False,
                    )
            finally:
                torch.load = _orig_torch_load  # type: ignore[assignment]

            # success
            break
        except Exception as e:
            last_error = e
            logger.warning(
                f"Model load failed on device [{device}], falling back. Error: {e}"
            )
            model = None
            continue

    if model is None:
        raise RuntimeError(
            f"Failed to load model on devices {devices_to_try}"
        ) from last_error

    model.conf = 0.250
    model.eval()
    app.state.model = model
    logger.info(f"Model loading for pid [{pid}] finished.")


"""Model Stats
      Class      P      R  mAP@.5 mAP@.5:.95                                                                                                                                                                                   
        all  0.955  0.825  0.895  0.676                                                                                                                                                                                                                                  
handwritten  0.961  0.766  0.843  0.619                                                                                                                                                                                                                                  
    typeset   0.95  0.883  0.948  0.733 
    
Best F1 score (all classes) at confidence 0.250
"""
