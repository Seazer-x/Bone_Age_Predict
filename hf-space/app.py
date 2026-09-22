import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import gradio as gr
import numpy as np
import requests
import spaces

from bone_age.bone_age import Bone_Age

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ["Radius", "Ulna", "MCPFirst", "MCP", "PIP", "PIPFirst", "MIP", "DIP", "DIPFirst"]
PART_WEIGHTS = [ROOT / "bone_age" / name / "best.pt" for name in MODEL_NAMES]
DETECTOR_WEIGHT = ROOT / "bone_age" / "bone_age.pt"

WEIGHT_URLS = {
    DETECTOR_WEIGHT: "https://raw.githubusercontent.com/Seazer-x/Bone_Age_Predict/v1.0.0/bone_age/bone_age.pt",
}
for name, path in zip(MODEL_NAMES, PART_WEIGHTS):
    WEIGHT_URLS[path] = (
        f"https://raw.githubusercontent.com/Seazer-x/Bone_Age_Predict/v1.0.0/"
        f"bone_age/{name}/best.pt"
    )


def ensure_weights():
    for path, url in WEIGHT_URLS.items():
        if path.exists() and path.stat().st_size > 1_000_000:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {path.name} -> {path}")
        with requests.get(url, stream=True, timeout=(15, 600)) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp.replace(path)


ensure_weights()

# ZeroGPU supports CUDA placement at module import time through CUDA emulation.
MODEL = Bone_Age(PART_WEIGHTS, MODEL_NAMES, device="0", fp16=True)
MODEL._load_detector(DETECTOR_WEIGHT)


@spaces.GPU(duration=120)
def predict(image, sex, confidence, iou):
    if image is None:
        return "Please upload a frontal single-hand X-ray image."

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

demo.queue().launch()
