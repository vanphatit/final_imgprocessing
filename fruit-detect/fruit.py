import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def show():
    # === ĐƯỜNG DẪN
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "fruits_yolov8n.onnx")  # Đổi nếu cần

    # === LOAD YOLO
    model_yolo = YOLO(MODEL_PATH, task="detect")

    # === LOAD IMAGE
    def load_image(uploaded_file):
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr

    # === DETECT FRUIT
    def detect_fruit(imgin):
        imgout = imgin.copy()
        annotator = Annotator(imgout)
        results = model_yolo.predict(imgin, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.tolist()
        names = model_yolo.names
        for box, cls, conf in zip(boxes, clss, confs):
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(box, label=label, txt_color=(255, 255, 255), color=(0, 153, 255))
        return imgout

    # === UI SETUP
    st.markdown(
        """
        <h1 style='text-align: center; color: #2c3e50;'>
            <img src='https://cdn-icons-png.flaticon.com/512/415/415733.png' width='48' style='vertical-align: middle; margin-bottom: 6px;'/>
            Nhận diện trái cây bằng YOLOv8
        </h1>
        """,
        unsafe_allow_html=True
    )

    mode = st.radio("🎯 Chọn chế độ:", ["Ảnh tĩnh", "Webcam real-time"], horizontal=True)

    # === ẢNH TĨNH
    if mode == "Ảnh tĩnh":
        uploaded_file = st.file_uploader("📂 Tải ảnh trái cây", type=["jpg", "jpeg", "png", "bmp"])

        if uploaded_file:
            img_color = load_image(uploaded_file)
            result_img = detect_fruit(img_color)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🖼️ **Ảnh Gốc**")
                st.image(img_color, channels="BGR", use_column_width=True)
            with col2:
                st.markdown("🔍 **Kết Quả Nhận Diện**")
                st.image(result_img, channels="BGR", use_column_width=True)
        else:
            st.info("📥 Hãy chọn một ảnh để bắt đầu nhận diện.")

    # === WEBCAM
    else:
        st.markdown("📸 **Bật webcam để nhận diện real-time**")
        run = st.checkbox("🎥 Bật webcam")
        frame_placeholder = st.image([])

        if run:
            cap = cv2.VideoCapture(0)
            while run and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result_frame = detect_fruit(frame)
                frame_placeholder.image(result_frame, channels="BGR")
            cap.release()
