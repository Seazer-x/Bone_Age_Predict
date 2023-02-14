from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from bone_age.bone_age import Bone_Age

msg = ""
TITLE = ""

st.title("Yolov5 Bone Age PreDict")

col1, col2 = st.columns(2, gap="large")
with col1:
    conf_threshold = st.slider("**Confidence Threshold**", 0.2, 1.0, 0.6, 0.01)
    iou_threshold = st.slider("**IOU Threshold**", 0.0, 1.0, 0.45, 0.01)
    sex = st.radio("**Sex**", ["boy", "girl"])
with col2:
    models = ["Radius", "Ulna", "MCPFirst", "MCP", "PIP",
              "PIPFirst", "MIP", "DIP", "DIPFirst"]
    weights = list(map(lambda x: Path("./bone_age/" + x + "/best.pt"), models))
    Bone = Bone_Age(weights, models)
    st.subheader('Input a Image')
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'], )
    if uploaded_file is not None:
        if TITLE != "" and msg != "":
            msg = ""
            TITLE = ""
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_cv2 = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        msg, success = Bone.run(weights_path=Path("./bone_age/bone_age.pt"), sex=sex, im=image_cv2,
                                conf_thres=conf_threshold, iou_thres=iou_threshold)
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
