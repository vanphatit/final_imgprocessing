import os
import numpy as np
import cv2 as cv
import joblib
import streamlit as st

def show():
    st.markdown("## 👤 Face Recognition (ONNX + SVM)")

    BASE_DIR = os.path.dirname(__file__)
    model_fd = os.path.join(BASE_DIR, 'face_detection_yunet_2023mar.onnx')
    model_fr = os.path.join(BASE_DIR, 'face_recognition_sface_2021dec.onnx')
    model_svc = os.path.join(BASE_DIR, 'svc.pkl')

    if not os.path.exists(model_svc):
        st.error(f"❌ Missing `svc.pkl` at: {model_svc}")
        return

    mydict = ['Chi Thanh', 'Nhu Quynh', 'Thanh Duy', 'Van Phat']
    svc = joblib.load(model_svc)

    detector = cv.FaceDetectorYN.create(model_fd, "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv.FaceRecognizerSF.create(model_fr, "")

    def predict_faces(frame):
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        faces = detector.detect(frame)
        predictions = []

        if faces[1] is not None:
            for face in faces[1]:
                aligned = recognizer.alignCrop(frame, face)
                embedding = recognizer.feature(aligned).flatten()
                pred_idx = svc.predict([embedding])[0]
                label = mydict[pred_idx] if pred_idx < len(mydict) else f"Unknown {pred_idx}"
                predictions.append((face, label))
        return frame, predictions

    def draw_faces(img, predictions):
        for face, label in predictions:
            coords = face[:-1].astype(np.int32)
            cv.rectangle(img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
            cv.putText(img, label, (coords[0], coords[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return img

    # ==== TABS ====
    tab1, tab2 = st.tabs(["📷 Webcam", "🖼️ Upload Image"])

    # ==== TAB 1 ====
    with tab1:
        run_cam = st.checkbox("📸 Turn on webcam")
        FRAME_WINDOW = st.image([])

        if run_cam:
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open webcam")
            else:
                st.warning("🔄 Pls uncheck 'Turn on webcam' button before switching to another feature.")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Cannot read frame")
                        break
                    frame, predictions = predict_faces(frame)
                    frame = draw_faces(frame, predictions)
                    FRAME_WINDOW.image(frame, channels="BGR")
                cap.release()
                cv.destroyAllWindows()

    # ==== TAB 2 ====
    with tab2:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv.imdecode(file_bytes, 1)  # BGR
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption="Original", use_container_width=True)

            img, predictions = predict_faces(img)

            if predictions:
                img = draw_faces(img, predictions)
                with col2:
                    st.image(img, caption="Recognition", channels="BGR", use_container_width=True)
            else:
                st.warning("😕 No faces detected in the image.")
