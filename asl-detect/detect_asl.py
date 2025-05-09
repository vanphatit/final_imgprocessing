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
MODEL_PATH = "asl-detect/asl-detector-retrained-v4.h5"
CLASS_NAMES_PATH = "asl-detect/class_names.txt"
BEEP_AUDIO_PATH = "asl-detect/ding.mp3"
MOVEMENT_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 0.7

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
prev_landmarks = None

def show():
    global prev_landmarks

    st.markdown("""
        <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 3em;
            font-weight: 800;
            margin-top: 0.5em;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #555;
            margin-bottom: 2em;
        }
        .detected-label {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 0.5em;
        }
        .detected-box {
            background-color: #f1f8ff;
            border: 3px solid #b3d1ff;
            border-radius: 12px;
            padding: 1.5em;
            font-size: 2em;
            font-weight: bold;
            color: #08415C;
            text-align: center;
            min-height: 250px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>🖐️ ASL Sign Language Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Show your hand sign to the webcam and watch the recognition in real time.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")
    frame_window = col1.empty()

    with col2:
        st.markdown("<div class='detected-label'>✅ Detected Characters</div>", unsafe_allow_html=True)
        recognized_text = st.markdown("<div class='detected-box'>...</div>", unsafe_allow_html=True)

    prediction_history = deque(maxlen=5)
    recognized_sequence = []
    current_label_overlay = ""

    def is_hand_stable(prev, curr):
        if not prev or not curr:
            return False
        return np.linalg.norm(np.array(prev) - np.array(curr)) < MOVEMENT_THRESHOLD

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
                if current_label_overlay:
                    cv2.putText(frame, current_label_overlay, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        return frame

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame)

        if prediction_history:
            top = Counter(prediction_history).most_common(1)[0][0]
            if not recognized_sequence or top != recognized_sequence[-1]:
                recognized_sequence.append(top)
                play_sound()

        html = f"<div class='detected-box'>{' '.join(recognized_sequence)}</div>"
        recognized_text.markdown(html, unsafe_allow_html=True)
        frame_window.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    cap.release()
