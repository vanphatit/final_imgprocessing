import os
import numpy as np
import cv2 as cv
import joblib
import streamlit as st

def show():
    # ==== Setup ====
    st.title("👤 Nhận diện khuôn mặt (ONNX + SVM)")
    FRAME_WINDOW = st.image([])

    # ==== Load Models ====
    BASE_DIR = os.path.dirname(__file__)  # chính là face-detect/

    model_fd = os.path.join(BASE_DIR, 'face_detection_yunet_2023mar.onnx')
    model_fr = os.path.join(BASE_DIR, 'face_recognition_sface_2021dec.onnx')
    model_svc = os.path.join(BASE_DIR, 'svc.pkl')



    if not os.path.exists(model_svc):
        st.error(f"❌ Không tìm thấy file `svc.pkl` tại: {model_svc}")
        st.stop()

    svc = joblib.load(model_svc)
    mydict = ['Chi Thanh', 'Nhu Quynh', 'Thanh Duy', 'Van Phat']

    detector = cv.FaceDetectorYN.create(model_fd, "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv.FaceRecognizerSF.create(model_fr, "")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Không mở được webcam")
        st.stop()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize((frameWidth, frameHeight))

    # ==== Hàm vẽ kết quả ====
    def visualize(input, faces, fps, predictions=None, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                for i in range(5):
                    cv.circle(input, (coords[4+i*2], coords[5+i*2]), 2, (0, 255, 255), thickness)
                if predictions:
                    label = predictions[idx]
                    cv.putText(input, label, (coords[0], coords[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(input, f'FPS: {fps:.2f}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ==== Main loop ====
    tm = cv.TickMeter()
    stop = st.button("🛑 Stop camera")

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Không đọc được frame từ webcam")
            break

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        predictions = []
        if faces[1] is not None:
            for face in faces[1]:
                aligned = recognizer.alignCrop(frame, face)
                embedding = recognizer.feature(aligned).flatten()
                pred_idx = svc.predict([embedding])[0]
                label = mydict[pred_idx] if pred_idx < len(mydict) else f"Unknown {pred_idx}"
                predictions.append(label)

        visualize(frame, faces, tm.getFPS(), predictions)
        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
    cv.destroyAllWindows()
