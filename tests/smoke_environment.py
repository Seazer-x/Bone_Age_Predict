"""Exercise installed application dependencies without weights or GPU startup.

Run separately from the lightweight regression suite in each complete environment.
Never import hf-space/app.py here: its module body downloads weights and starts CUDA
and a server. The Space check below validates CPU compatibility, not ZeroGPU.
"""

import argparse
import ast
import importlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_common():
    import cv2
    import numpy as np
    import torch
    import torchvision

    for module in (
        "IPython", "matplotlib", "pandas", "yaml", "scipy", "seaborn",
        "psutil", "thop", "tqdm", "requests", "packaging", "pkg_resources",
        "utils.general", "utils.augmentations", "utils.dataloaders",
        "utils.torch_utils", "models.common", "models.experimental", "models.yolo",
        "bone_age.bone_age", "evaluation.evaluate_dataset",
    ):
        print(f"Importing {module}", flush=True)
        importlib.import_module(module)

    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    tensor = torch.from_numpy(values)
    np.testing.assert_array_equal(tensor.numpy(), values)
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0, 2]

    from utils.augmentations import classify_transforms, letterbox
    image = np.arange(96 * 80 * 3, dtype=np.uint8).reshape(96, 80, 3)
    assert cv2.resize(image, (32, 32)).shape == (32, 32, 3)
    assert letterbox(image, (64, 64), auto=False)[0].shape == (64, 64, 3)
    assert classify_transforms(64)(image).shape == (3, 64, 64)
    print(f"Common smoke checks passed: torch={torch.__version__}, torchvision={torchvision.__version__}")
    return image


def check_local(image):
    import numpy as np
    import streamlit as st
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from utils.augmentations import Albumentations, classify_albumentations

    for module in ("albumentations", "skimage", "sklearn", "qudida"):
        importlib.import_module(module)

    detection = Albumentations(size=64)
    assert detection.transform is not None, "Detection augmentation was silently disabled"
    assert detection.transform.transforms, "Detection augmentation is empty"
    labels = np.array([[0., 0.5, 0.5, 0.25, 0.25]], dtype=np.float32)
    augmented, labels = detection(image, labels, p=1.0)
    assert augmented.shape == image.shape and labels.shape == (1, 5)
    assert np.isfinite(labels).all()
    for augment in (True, False):
        transform = classify_albumentations(augment=augment, size=64)
        assert transform is not None, "Classification augmentation was silently disabled"
        assert transform.transforms, "Classification augmentation is empty"
        result = transform(image=image)["image"]
        assert result.shape == (3, 64, 64) and torch.isfinite(result).all()

    # Execute the actual image call's keywords against the installed Streamlit API.
    # Reading its AST avoids starting the app or loading model weights.
    tree = ast.parse((ROOT / "Bone-pre.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "st" and node.func.attr == "image"]
    assert calls, "No Streamlit image call found"
    for call in calls:
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg != "caption"}
        st.image(image, caption="Dependency smoke check", **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        with SummaryWriter(directory) as writer:
            writer.add_scalar("smoke/value", 1.0, 0)
        assert list(Path(directory).glob("events.out.tfevents.*"))
    print(f"Local smoke checks passed: streamlit={st.__version__}; augmentation and TensorBoard exercised")


def check_space():
    import gradio as gr
    import spaces

    # Build representative components without launching or allocating a GPU.
    with gr.Blocks() as demo:
        image = gr.Image(type="numpy")
        sex = gr.Radio(["boy", "girl"], value="boy")
        confidence = gr.Slider(0.2, 1.0, value=0.4)
        iou = gr.Slider(0.0, 1.0, value=0.45)
        output = gr.Textbox()
        gr.Button("Run").click(lambda *args: "smoke", [image, sex, confidence, iou], output)
    assert demo.config["components"]
    assert callable(spaces.GPU)
    print(f"Space CPU smoke checks passed: gradio={gr.__version__}; ZeroGPU requires hosted deployment verification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("local", "space"), required=True)
    args = parser.parse_args()
    image = check_common()
    if args.target == "local":
        check_local(image)
    else:
        check_space()


if __name__ == "__main__":
    main()
