import streamlit as st
import os
import sys

# Cho phép import từ face-detect/
st.set_page_config(page_title="AI Image App", layout="wide")
sys.path.append(os.path.join(os.path.dirname(__file__), "face-detect"))
from predict import show as show_face

st.sidebar.title("📂 Menu chức năng")

menu = st.sidebar.radio("Chọn chức năng", [
    "👤 Nhận diện khuôn mặt",
    # (Thêm các tab khác sau này)
])

if menu == "👤 Nhận diện khuôn mặt":
    show_face()
