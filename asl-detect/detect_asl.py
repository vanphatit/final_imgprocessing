import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit.components.v1 as components
from collections import deque, Counter
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
import base64

# ==== CONFIG ====
MODEL_PATH = "asl-detect/asl-detector-retrained-v3.h5"
CLASS_NAMES_PATH = "asl-detect/class_names.txt"
BEEP_AUDIO_PATH = "asl-detect/ding.mp3"
MOVEMENT_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 0.7

# ==== LOAD MODEL & LABELS ====
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ==== MEDIA PIPE ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ==== STREAMLIT UI ====
st.set_page_config(page_title="ASL Detection", layout="centered")
st.markdown("""
    <style>
    .centered { text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='centered'>🖐️ ASL Sign Language Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='centered'>Use your webcam to detect sign language characters.</p>", unsafe_allow_html=True)

# ==== Button + Text in same row ====
col1, col2 = st.columns([1, 2])
start_button = col1.button("▶️ Start Detection")
recognized_text = col2.empty()

# ==== Webcam view ====
frame_window = st.image([], width=480)

# ==== STATE ====
prediction_history = deque(maxlen=5)
recognized_sequence = []
prev_landmarks = None
current_label_overlay = ""

# ==== UTILS ====
def is_hand_stable(prev, curr):
    if not prev or not curr:
        return False
    diff = np.linalg.norm(np.array(prev) - np.array(curr))
    return diff < MOVEMENT_THRESHOLD

def play_sound():
    audio_path = Path(BEEP_AUDIO_PATH)
    if audio_path.exists():
        audio_bytes = audio_path.read_bytes()
        b64 = base64.b64encode(audio_bytes).decode()
        components.html(f"""
            <audio autoplay hidden>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """, height=0)

def process_frame(frame):
    global prev_landmarks, current_label_overlay
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    current_label_overlay = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue
            hand_img = cv2.resize(hand_img, (224, 224))
            hand_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            input_img = preprocess_input(np.expand_dims(hand_rgb, axis=0))

            current_landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            if is_hand_stable(prev_landmarks, current_landmarks):
                pred = model.predict(input_img, verbose=0)
                confidence = np.max(pred[0])
                class_id = np.argmax(pred[0])
                if confidence > CONFIDENCE_THRESHOLD:
                    label = labels[class_id].upper()
                    prediction_history.append(label)
                    current_label_overlay = f"{label} ({confidence * 100:.0f}%)"

            prev_landmarks = current_landmarks

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # === Hiện ký tự + độ chính xác trực tiếp trên hình
            if current_label_overlay:
                cv2.putText(frame, current_label_overlay, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return frame

# ==== MAIN LOOP ====
if start_button:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        # Cập nhật dòng ký tự bên cạnh nút
        if prediction_history:
            top_label = Counter(prediction_history).most_common(1)[0][0]
            if not recognized_sequence or top_label != recognized_sequence[-1]:
                recognized_sequence.append(top_label)
                play_sound()

        html_output = f"<h4 style='color:#228B22;'>{' '.join(recognized_sequence)}</h4>"
        recognized_text.markdown(html_output, unsafe_allow_html=True)

        frame_window.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

    cap.release()
