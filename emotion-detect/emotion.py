import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import os

# === Load model ===
@st.cache_resource(show_spinner="🔄 Đang tải mô hình nhận diện cảm xúc...")
def load_emotion_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "emotion_model.h5")
    return load_model(model_path)

# === Phát hiện và cắt khuôn mặt ===
def detect_and_crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    # Lấy khuôn mặt lớn nhất
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_img = image[y:y+h, x:x+w]
    return face_img

# === Dự đoán cảm xúc ===
def predict_emotion(image_bgr, model):
    face = detect_and_crop_face(image_bgr)
    if face is None:
        return "Không tìm thấy khuôn mặt", 0.0, "❌"

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emoji_map = {
        "Angry": "😠", "Disgust": "🤢", "Fear": "😱", "Happy": "😄",
        "Sad": "😢", "Surprise": "😲", "Neutral": "😐"
    }

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    norm = resized.astype('float32') / 255.0
    input_tensor = np.expand_dims(norm, axis=(0, -1))  # (1, 48, 48, 1)
    preds = model.predict(input_tensor, verbose=0)[0]
    label_idx = np.argmax(preds)
    return emotion_labels[label_idx], preds[label_idx], emoji_map[emotion_labels[label_idx]]

# === Giao diện chính ===
def show_emotion_app():
    st.title("🧠 Nhận diện Cảm xúc qua Khuôn mặt")
    st.markdown("Sử dụng mô hình học sâu để phân tích biểu cảm khuôn mặt qua ảnh hoặc webcam.")

    model = load_emotion_model()
    mode = st.radio("🎛️ Chọn chế độ:", ["📁 Tải ảnh", "📸 Chụp ảnh webcam", "🎥 Webcam real-time"], horizontal=True)

    if mode == "📁 Tải ảnh":
        uploaded_file = st.file_uploader("📤 Tải ảnh khuôn mặt", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = np.array(Image.open(uploaded_file))
            st.image(img, caption="🖼️ Ảnh đã tải", use_column_width=True)

            label, conf, emoji = predict_emotion(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), model)
            st.markdown("---")
            st.markdown(f"### 🔍 Dự đoán: **{label}** {emoji}")
            st.markdown(f"#### 🎯 Độ tin cậy: `{conf * 100:.2f}%`")

    elif mode == "📸 Chụp ảnh webcam":
        img_file = st.camera_input("📸 Chụp ảnh khuôn mặt")
        if img_file:
            img = np.array(Image.open(img_file))
            st.image(img, caption="📷 Ảnh vừa chụp", use_container_width=True)

            label, conf, emoji = predict_emotion(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), model)
            st.markdown("---")
            st.markdown(f"### 🔍 Dự đoán: **{label}** {emoji}")
            st.markdown(f"#### 🎯 Độ tin cậy: `{conf * 100:.2f}%`")

    elif mode == "🎥 Webcam real-time":
        run = st.checkbox("▶️ Bắt đầu phát hiện real-time", key="run_webcam")
        placeholder = st.empty()

        if run:
            st.warning("⏹️ Bỏ tick để dừng webcam.")
            cap = cv2.VideoCapture(0)

            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("🚫 Không thể truy cập webcam.")
                    break

                frame = cv2.flip(frame, 1)
                label, conf, emoji = predict_emotion(frame, model)

                cv2.putText(frame, f"{label} {emoji} ({conf*100:.1f}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder.image(frame_rgb, channels="RGB")

            cap.release()
            st.success("✅ Đã dừng webcam.")

