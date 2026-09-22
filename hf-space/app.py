import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import gradio as gr
import numpy as np
import requests
import spaces
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bone_age.bone_age import Bone_Age

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = Path(
    os.getenv(
        "BONE_AGE_CACHE_DIR",
        str(Path.home() / ".cache" / "bone-age-predict" / "v1.0.0"),
    )
)

MODEL_NAMES = [
    "Radius",
    "Ulna",
    "MCPFirst",
    "MCP",
    "PIP",
    "PIPFirst",
    "MIP",
    "DIP",
    "DIPFirst",
]

WEIGHT_SPECS = {
    "bone_age/bone_age.pt": 92_730_597,
    "bone_age/Radius/best.pt": 50_784_548,
    "bone_age/Ulna/best.pt": 50_779_428,
    "bone_age/MCPFirst/best.pt": 50_776_868,
    "bone_age/MCP/best.pt": 50_774_244,
    "bone_age/PIP/best.pt": 50_779_428,
    "bone_age/PIPFirst/best.pt": 50_779_428,
    "bone_age/MIP/best.pt": 50_779_428,
    "bone_age/DIP/best.pt": 50_776_804,
    "bone_age/DIPFirst/best.pt": 50_776_868,
}

BASE_URL = "https://raw.githubusercontent.com/Seazer-x/Bone_Age_Predict/v1.0.0"
DETECTOR_WEIGHT = CACHE_ROOT / "bone_age" / "bone_age.pt"
PART_WEIGHTS = [CACHE_ROOT / "bone_age" / name / "best.pt" for name in MODEL_NAMES]
EXAMPLE_IMAGE = ROOT / "examples" / "1778216012401.jpg"


def _http_session():
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    return session


def _download_weight(relative_path: str, expected_size: int) -> Path:
    target = CACHE_ROOT / relative_path
    if target.exists() and target.stat().st_size == expected_size:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)

    url = f"{BASE_URL}/{relative_path}"
    session = _http_session()
    try:
        with session.get(url, stream=True, timeout=(15, 600)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    finally:
        session.close()

    actual_size = partial.stat().st_size if partial.exists() else 0
    if actual_size != expected_size:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Weight download size mismatch for {relative_path}: "
            f"expected={expected_size}, actual={actual_size}"
        )

    partial.replace(target)
    return target


def ensure_weights() -> None:
    missing = [
        (relative_path, size)
        for relative_path, size in WEIGHT_SPECS.items()
        if not (
            (CACHE_ROOT / relative_path).exists()
            and (CACHE_ROOT / relative_path).stat().st_size == size
        )
    ]
    if not missing:
        print(f"Using cached model weights from {CACHE_ROOT}")
        return

    print(f"Downloading {len(missing)} model files to cache: {CACHE_ROOT}")
    with ThreadPoolExecutor(max_workers=min(4, len(missing))) as pool:
        futures = {
            pool.submit(_download_weight, relative_path, size): relative_path
            for relative_path, size in missing
        }
        for future in as_completed(futures):
            path = future.result()
            print(f"Ready: {path}")


ensure_weights()

# ZeroGPU supports CUDA placement at module import time through CUDA emulation.
MODEL = Bone_Age(PART_WEIGHTS, MODEL_NAMES, device="0", fp16=True)
MODEL._load_detector(DETECTOR_WEIGHT)


@spaces.GPU(duration=120)
def predict(image, sex, confidence, iou):
    if image is None:
        return "Please upload a frontal single-hand X-ray image or choose the example below."

    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    message, success = MODEL.run(
        weights_path=DETECTOR_WEIGHT,
        sex=sex,
        im=image,
        conf_thres=float(confidence),
        iou_thres=float(iou),
    )
    return message if success else f"⚠️ {message}"


with gr.Blocks(title="Bone Age Predict") as demo:
    gr.Markdown(
        """
# 🦴 Bone Age Predict
YOLOv5 + RUS-CHN bone age estimation demo.

> **Research and educational use only.** This is not a medical device and must not be used as an independent clinical diagnosis.

[GitHub repository](https://github.com/Seazer-x/Bone_Age_Predict)
"""
    )

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="numpy", label="Hand X-ray")
            sex = gr.Radio(["boy", "girl"], value="boy", label="RUS-CHN scoring sex")
            confidence = gr.Slider(0.20, 1.00, value=0.40, step=0.01, label="Detection confidence")
            iou = gr.Slider(0.00, 1.00, value=0.45, step=0.01, label="IoU threshold")
            run = gr.Button("Run inference", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Bone age report", lines=18)

    run.click(
        fn=predict,
        inputs=[image, sex, confidence, iou],
        outputs=output,
    )

    if EXAMPLE_IMAGE.exists():
        gr.Markdown("### Example")
        gr.Examples(
            examples=[[str(EXAMPLE_IMAGE), "boy", 0.40, 0.45]],
            inputs=[image, sex, confidence, iou],
            outputs=output,
            fn=predict,
            cache_examples=False,
            label="Click the sample X-ray to load it",
        )


demo.queue().launch()
