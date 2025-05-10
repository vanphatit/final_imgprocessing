import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd

def show():
    st.title("🖼️ YOLOv8 Image Detection")
    st.markdown("Nhận diện đối tượng từ ảnh bằng mô hình **YOLOv8** đã huấn luyện.")

    # Đường dẫn model và threshold mặc định
    model_path = "obj-detect/yolov8_coco_trained.pt"
    conf = 0.3

    # Load model YOLO
    @st.cache_resource
    def load_model(path):
        return YOLO(path)

    model = load_model(model_path)

    # 🔍 Thông tin model
    st.subheader("🧠 Thông tin mô hình")
    st.markdown(f"- **Tên mô hình**: `{model_path.split('/')[-1]}`")
    st.markdown(f"- **Số lượng lớp**: `{len(model.names)}` lớp")
    
    with st.expander("📚 Danh sách lớp mà model có thể nhận diện"):
        cols = st.columns(4)
        for i, name in enumerate(model.names.values()):
            with cols[i % 4]:
                st.markdown(f"- {name}")

    # Upload ảnh
    uploaded_file = st.file_uploader("📤 Upload một ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.subheader("🖼️ Ảnh gốc")
        st.image(img_array, use_column_width=True)

        # Nhận diện
        results = model.predict(source=img_array, conf=conf)

        # Vẽ bounding boxes
        res_plotted = results[0].plot()
        st.subheader("🎯 Kết quả nhận diện")
        st.image(res_plotted, use_column_width=True)

        # Lấy kết quả boxes
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        if len(class_ids) == 0:
            st.info("🤷 Không phát hiện được đối tượng nào.")
            return

        # Tạo bảng thống kê
        labels = [model.names[i] for i in class_ids]
        df = pd.DataFrame({
            "Label": labels,
            "Confidence": confidences
        })

        summary = df.groupby("Label").agg(
            Count=("Label", "count"),
            AvgConfidence=("Confidence", lambda x: round(x.mean(), 2))
        ).reset_index().sort_values(by="Count", ascending=False)

        st.subheader("📊 Thống kê nhãn phát hiện được")
        st.dataframe(summary, use_container_width=True)
