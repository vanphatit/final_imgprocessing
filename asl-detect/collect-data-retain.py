import cv2
import os
import mediapipe as mp
from datetime import datetime

# Tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Tạo thư mục nếu chưa có
all_labels = [chr(c) for c in range(ord("a"), ord("z") + 1) if chr(c) != "j"]
for label in all_labels:
    os.makedirs(f"asl-detect/dataset_retrain/{label}", exist_ok=True)

print("👉 Bấm phím a–z để lưu ảnh, q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Bounding box tay
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            # Crop + resize
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue
            hand_img = cv2.resize(hand_img, (224, 224))

            # Hiển thị + landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Hiển thị
    cv2.imshow("Collect ASL A-Z (q to quit)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif 97 <= key <= 122 and chr(key) != "j":  # a–z bỏ j
        label = chr(key)
        if results.multi_hand_landmarks:
            filename = f"asl-detect/dataset_retrain/{label}/{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
            cv2.imwrite(filename, hand_img)
            print(f"✅ Saved {label.upper()} → {filename}")

cap.release()
cv2.destroyAllWindows()
hands.close()
