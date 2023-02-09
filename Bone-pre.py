import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parent  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from bone_age.bone_age import Bone_Age

msg = ""
TITLE = ""
st.title("Yolov5 Bone Age PreDict")

col1, col2 = st.columns(2, gap="large")
with col1:
    conf_thres = st.slider("**Confidence Threshold**", 0.2, 1.0, 0.6, 0.01)
    iou_thres = st.slider("**IOU Threshold**", 0.0, 1.0, 0.45, 0.01)
    sex = st.radio("**Sex**", ["boy", "girl"])
with col2:
    models = ["Radius", "Ulna", "MCPFirst", "MCP", "PIP",
              "PIPFirst", "MIP", "DIP", "DIPFirst"]
    weights = list(map(lambda x: "bone_age/" + x + "/weights/best.pt", models))
    Bone = Bone_Age(weights, models)
    st.subheader('Input a Image')
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'], )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_cv2 = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        msg, success = Bone.run(weights_path="bone_age/bone_age.pt", sex=sex, im=image_cv2, conf_thres=conf_thres,
                                iou_thres=iou_thres)
        if not success:
            st.error("推理失败！")
            msg = ""
            TITLE = ""
        else:
            st.success("推理成功！")
            TITLE = "推理结果:"
    else:
        st.info('☝️ Upload a Image file')
st.subheader(TITLE)
st.text(msg)
