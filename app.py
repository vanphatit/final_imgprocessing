import streamlit as st
import os
import sys

# Cho phép import từ face-detect/
st.set_page_config(page_title="AI Image App", layout="wide")
sys.path.append(os.path.join(os.path.dirname(__file__), "face-detect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "fruit-detect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "img-process"))
sys.path.append(os.path.join(os.path.dirname(__file__), "emotion-detect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "skin-resnet18"))

from predict import show as show_face
from fruit import show as show_fruit
from processing import show as show_process
from emotion import show_emotion_app as show_emotion
from skin_resnet18 import show_skin as show_skin

st.sidebar.title("📂 Menu chức năng")

menu = st.sidebar.radio("Chọn chức năng", [
    "👤 Nhận diện khuôn mặt",
    "🍎 Nhận diện trái cây",
    "🖼️ Xử lý ảnh",
    "😊 Nhận diện cảm xúc",
    "🩺 Nhận diện bệnh da liễu (HAM10000)",
    # (Thêm các tab khác sau này)
])

if menu == "👤 Nhận diện khuôn mặt":
    show_face()
elif menu == "🍎 Nhận diện trái cây":
    show_fruit()
elif menu == "🖼️ Xử lý ảnh":
    show_process()
elif menu == "😊 Nhận diện cảm xúc":
    show_emotion()
elif menu == "🩺 Nhận diện bệnh da liễu (HAM10000)":
    show_skin()