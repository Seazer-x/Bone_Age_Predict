from pathlib import Path
import ast
import os
import pathlib
import re

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


st.set_page_config(
    page_title="Bone Age Predict",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1240px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }

    .main-title {
        font-size: 2.15rem;
        line-height: 1.2;
        font-weight: 760;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }

    .main-subtitle {
        color: #667085;
        font-size: 0.98rem;
        margin-bottom: 1.25rem;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }

    .param-card {
        border: 1px solid #EAECF0;
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
        background: #FCFCFD;
        margin-bottom: 0.85rem;
    }

    .param-row {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        font-size: 0.92rem;
        margin: 0.25rem 0;
    }

    .param-label {
        color: #667085;
    }

    .param-value {
        color: #101828;
        font-weight: 700;
    }

    .empty-panel {
        border: 1px dashed #D0D5DD;
        border-radius: 16px;
        padding: 3rem 1.25rem;
        text-align: center;
        color: #667085;
        background: #FCFCFD;
    }

    div[data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid #EAECF0;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
    }

    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }

    .sidebar-caption {
        color: #667085;
        font-size: 0.86rem;
        margin-bottom: 1rem;
    }

    .small-note {
        color: #667085;
        font-size: 0.85rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def split_inference_message(message: str):
    text = (message or "").strip()
    text = re.sub(r"^推理失败[:：]\s*", "", text)
    parts = [part.strip() for part in re.split(r"\s*；\s*", text) if part.strip()]
    if not parts:
        return "未知错误", [], text
    return parts[0], parts[1:], text


def render_detail_item(detail: str):
    if detail.startswith("当前检测计数："):
        raw_counts = detail.split("：", 1)[1].strip()
        try:
            counts = ast.literal_eval(raw_counts)
        except Exception:
            st.code(raw_counts, language="text")
            return

        rows = [{"类别": name, "检测数量": count} for name, count in counts.items()]
        st.markdown("**当前检测计数**")
        st.table(rows)
        return

    if detail.startswith("缺失/不足："):
        missing = detail.split("：", 1)[1].strip()
        items = [item.strip() for item in missing.split(";") if item.strip()]
        st.markdown("**缺失 / 不足**")
        for item in items:
            st.markdown(f"- `{item}`")
        return

    st.markdown(f"- {detail}")


def render_failure_message(message: str):
    summary, details, full_text = split_inference_message(message)
    st.error(f"推理失败：{summary}")

    if details:
        with st.expander("检测详情", expanded=False):
            for detail in details:
                render_detail_item(detail)

    with st.expander("原始错误文本", expanded=False):
        st.code(full_text or message or "无错误详情", language="text")


def render_success_message(message: str):
    st.success("推理成功")
    st.markdown('<div class="section-title">骨龄报告</div>', unsafe_allow_html=True)
    st.text(message)


def render_runtime_params(sex: str, conf_threshold: float, iou_threshold: float):
    st.markdown(
        f"""
        <div class="param-card">
            <div class="param-row">
                <span class="param-label">性别</span>
                <span class="param-value">{sex}</span>
            </div>
            <div class="param-row">
                <span class="param-label">置信度阈值</span>
                <span class="param-value">{conf_threshold:.2f}</span>
            </div>
            <div class="param-row">
                <span class="param-label">IOU 阈值</span>
                <span class="param-value">{iou_threshold:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="sidebar-title">🦴 Bone Age</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">YOLOv5 目标检测 + YOLOv5 分类模型</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "上传手部 X 光图像",
        type=["png", "jpg", "jpeg"],
        help="支持 png、jpg、jpeg。建议上传单手正位 X 光图。",
    )

    st.divider()
    st.markdown("### 推理参数")
    sex = st.radio("性别", ["boy", "girl"], horizontal=True)
    conf_threshold = st.slider("置信度阈值", 0.2, 1.0, 0.6, 0.01)
    iou_threshold = st.slider("IOU 阈值", 0.0, 1.0, 0.45, 0.01)

    st.divider()
    auto_run = st.checkbox("上传后自动推理", value=True)
    run_clicked = st.button(
        "开始推理",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    st.markdown(
        '<p class="small-note">检测失败时，建议先降低置信度阈值，例如 0.35 或 0.25。</p>',
        unsafe_allow_html=True,
    )


st.markdown('<div class="main-title">Yolov5 Bone Age Predict</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">上传手部 X 光图像，自动检测 RUS-CHN 评分区域并输出骨龄报告。</div>',
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="empty-panel">
            <h3>等待上传图像</h3>
            <p>请在左侧侧边栏上传 png/jpg/jpeg 格式的手部 X 光图像。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    image_cv2 = decode_uploaded_image(uploaded_file)
except Exception as exc:
    st.error("图片读取失败")
    st.exception(exc)
    st.stop()

image_col, result_col = st.columns([1.05, 1.15], gap="large")

with image_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">图像预览</div>', unsafe_allow_html=True)
        st.image(image_cv2, caption=uploaded_file.name, use_column_width=True)

with result_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">推理结果</div>', unsafe_allow_html=True)
        render_runtime_params(sex, conf_threshold, iou_threshold)

        should_run = auto_run or run_clicked
        if not should_run:
            st.info("已上传图像。点击左侧「开始推理」后运行模型。")
            st.stop()

        try:
            Bone = load_bone_age_model()
        except Exception as exc:
            st.error("模型加载失败")
            st.exception(exc)
            st.stop()

        with st.spinner("正在推理..."):
            try:
                msg, success = Bone.run(
                    weights_path=DETECTOR_WEIGHT,
                    sex=sex,
                    im=image_cv2,
                    conf_thres=conf_threshold,
                    iou_thres=iou_threshold,
                )
            except Exception as exc:
                st.error("推理异常")
                st.exception(exc)
                st.stop()

        if success:
            render_success_message(msg)
        else:
            render_failure_message(msg)
