import streamlit as st
import os
import sys

# Cho phép import từ face-detect/
st.set_page_config(page_title="AI Image App", layout="wide")
sys.path.append(os.path.join(os.path.dirname(__file__), "face-detect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "fruit-detect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "img-process"))

from predict import show as show_face
from fruit import show as show_fruit
from processing import show as show_process

st.sidebar.title("📂 Menu chức năng")

menu = st.sidebar.radio("Chọn chức năng", [
    "👤 Nhận diện khuôn mặt",
    "🍎 Nhận diện trái cây",
    "🖼️ Xử lý ảnh",
    # (Thêm các tab khác sau này)
])

if menu == "👤 Nhận diện khuôn mặt":
    show_face()
elif menu == "🍎 Nhận diện trái cây":
    show_fruit()
elif menu == "🖼️ Xử lý ảnh":
    show_process()