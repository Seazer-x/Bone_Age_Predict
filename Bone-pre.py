from pathlib import Path
import os
import pathlib

# Some YOLOv5 .pt files were saved on Windows. Streamlit Cloud runs on Linux,
# so patch WindowsPath before any model checkpoint is loaded.
if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath

import cv2
import numpy as np
import streamlit as st

from bone_age.bone_age import Bone_Age


ROOT = Path(__file__).resolve().parent
MODEL_NAMES = [
    "Radius", "Ulna", "MCPFirst", "MCP", "PIP",
    "PIPFirst", "MIP", "DIP", "DIPFirst",
]
PART_WEIGHTS = [ROOT / "bone_age" / name / "best.pt" for name in MODEL_NAMES]
DETECTOR_WEIGHT = ROOT / "bone_age" / "bone_age.pt"


@st.cache_resource(show_spinner="正在加载骨龄模型，请稍候...")
def load_bone_age_model():
    missing = [p for p in [*PART_WEIGHTS, DETECTOR_WEIGHT] if not p.exists()]
    if missing:
        raise FileNotFoundError("模型权重缺失：\n" + "\n".join(str(p) for p in missing))
    return Bone_Age(PART_WEIGHTS, MODEL_NAMES)


def decode_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("图片解码失败，请上传有效的 png/jpg/jpeg 图像。")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


st.title("Yolov5 Bone Age PreDict")

col1, col2 = st.columns(2, gap="large")
with col1:
    conf_threshold = st.slider("**Confidence Threshold**", 0.2, 1.0, 0.6, 0.01)
    iou_threshold = st.slider("**IOU Threshold**", 0.0, 1.0, 0.45, 0.01)
    sex = st.radio("**Sex**", ["boy", "girl"])

with col2:
    try:
        Bone = load_bone_age_model()
    except Exception as exc:
        st.error("模型加载失败。")
        st.exception(exc)
        st.stop()

    st.subheader("Input a Image")
    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

    title = ""
    msg = ""

    if uploaded_file is None:
        st.info("☝️ Upload a Image file")
    else:
        try:
            image_cv2 = decode_uploaded_image(uploaded_file)
            st.image(image_cv2, caption="Uploaded image", use_column_width=True)

            msg, success = Bone.run(
                weights_path=DETECTOR_WEIGHT,
                sex=sex,
                im=image_cv2,
                conf_thres=conf_threshold,
                iou_thres=iou_threshold,
            )

            if not success:
                st.error(msg or "推理失败！")
            else:
                st.success("推理成功！")
                title = "推理结果:"

        except Exception as exc:
            st.error("推理异常。")
            st.exception(exc)

st.subheader(title)
if msg:
    st.text(msg)
