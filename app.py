from streamlit_option_menu import option_menu
import streamlit as st
import os
import sys

# ==== CONFIG ====
st.set_page_config(page_title="Image Processing App - Van Phat & Thanh Duy", layout="wide")
BASE_DIR = os.path.dirname(__file__)
MODULES = ["face-detect", "fruit-detect", "img-process", "emotion-detect", "asl-detect", "obj-detect"]
for module in MODULES:
    sys.path.append(os.path.join(BASE_DIR, module))

# ==== MODULE IMPORTS ====
from home import show as show_home
from predict import show as show_face
from fruit import show as show_fruit
from processing import show as show_process
from emotion import show_emotion_app as show_emotion
from detect_asl import show as show_asl
from detect_obj import show as show_obj

# ==== MENU SETUP ====
MENU_ITEMS = {
    "Home": show_home,
    "Face Detection": show_face,
    "Fruit Recognition": show_fruit,
    "Image Processing": show_process,
    "Emotion Detection": show_emotion,
    "Sign Language": show_asl,
    "Object Detection": show_obj,
}
BOOTSTRAP_ICONS = [
    "house",                # Home
    "person-bounding-box",   # Face Detection
    "apple",                 # Fruit Recognition
    "images",                # Image Processing
    "emoji-smile",           # Emotion Detection
    "hand-index-thumb",      # Sign Language
    "box",                   # Object Detection
]

# ==== SIDEBAR ====
with st.sidebar:
    st.image("hcmute.png", use_container_width=True)

    selected = option_menu(
        menu_title="Features",
        options=list(MENU_ITEMS.keys()),
        icons=BOOTSTRAP_ICONS,
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#f0f2f6"},
            "icon": {"color": "#4F46E5", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "padding": "10px 16px",
                "margin": "4px 0",
                "--hover-color": "#eee",
                "font-weight": "bold"
            },
            "nav-link-selected": {
                "background-color": "#4F46E5",
                "color": "white",
            },
        }
    )

    st.markdown("""
    <div style="font-size: 14px;">
        <b>Team: Văn Phát & Thanh Duy</b><br>
        <ul style="padding-left: 0;">
            <li>Lê Văn Phát - 22110196</li>
            <li>Huỳnh Thanh Duy - 22110118</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==== MAIN ====
MENU_ITEMS[selected]()