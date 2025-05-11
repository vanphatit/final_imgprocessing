import streamlit as st

def show():
    st.markdown("""
        <style>
            .avatar {
                border-radius: 50%;
                width: 120px;
                height: 120px;
                object-fit: cover;
                border: 3px solid #4F46E5;
            }
            .name {
                font-size: 20px;
                font-weight: bold;
                margin-top: 0.5rem;
            }
            .mssv {
                font-size: 16px;
                color: #666;
            }
            .feature-box {
                background-color: #f3f4f6;
                padding: 16px;
                border-radius: 12px;
                margin-bottom: 12px;
                border-left: 5px solid #4F46E5;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌠 Image Processing Project: Multi-functional Recognition System")
    st.markdown("### 👨‍💻 Team Members")

    col1, col2 = st.columns(2)
    with col1:
        st.image("van-phat.jpg", caption="Lê Văn Phát", use_container_width=False, width=120)
        st.markdown("<div class='name'>Lê Văn Phát</div>", unsafe_allow_html=True)
        st.markdown("<div class='mssv'>MSSV: 22110196</div>", unsafe_allow_html=True)
    with col2:
        st.image("thanh-duy.jpg", caption="Huỳnh Thanh Duy", use_container_width=False, width=120)
        st.markdown("<div class='name'>Huỳnh Thanh Duy</div>", unsafe_allow_html=True)
        st.markdown("<div class='mssv'>MSSV: 22110118</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📌 Project Overview")
    st.markdown("""
        This project is an AI-powered multi-functional system that integrates:
        - Real-time webcam interaction
        - Deep learning + ONNX/TensorFlow models
        - Sign language (ASL), facial, object, fruit, and emotion recognition
        - Traditional image processing and enhancement
    """)

    st.markdown("### ⚙️ Main Features")

    features = [
    {
        "icon": "👤",
        "title": "Face Recognition",
        "desc": """
        <ul>
            <li>🔍 Detect faces using <code>YuNet (ONNX)</code></li>
            <li>🧠 Extract embeddings via <code>SFace (ONNX)</code></li>
            <li>🗂️ Classify identity using pretrained <code>SVM (joblib)</code></li>
            <li>🖥️ Input: Webcam or uploaded photo</li>
        </ul>
        """
    },
    {
        "icon": "😊",
        "title": "Emotion Detection",
        "desc": """
        <ul>
            <li>🎭 Recognize 7 basic emotions</li>
            <li>🧠 Model: CNN (trained on FER2013)</li>
            <li>📍 Use MediaPipe FaceMesh for keypoint extraction</li>
            <li>📷 Real-time webcam input</li>
        </ul>
        """
    },
    {
        "icon": "🤟",
        "title": "ASL Sign Language Detection",
        "desc": """
        <ul>
            <li>✋ Detect hand landmarks using MediaPipe</li>
            <li>📦 Model: MobileNetV2 retrained on our own ASL dataset <code>dataset_retrain</code></li>
            <li>🧠 Predict signs A–Z except J & Q using CNN</li>
            <li>📷 Input: Webcam (real-time)</li>
        </ul>
        """
    },
    {
        "icon": "📦",
        "title": "Object Detection",
        "desc": """
        <ul>
            <li>🚀 YOLOv8 pretrained model</li>
            <li>🎯 Detect objects from COCO classes</li>
            <li>🖼️ Input: static image</li>
        </ul>
        """
    },
    {
        "icon": "🍎",
        "title": "Fruit Classification",
        "desc": """
        <ul>
            <li>🍌 Identify fruits (apple, banana, orange...)</li>
            <li>📦 Model: Yolov8 </li>
            <li>🖼️ Input: uploaded image</li>
        </ul>
        """
    },
    {
        "icon": "🖼️",
        "title": "Image Processing",
        "desc": """
        <ul>
            <li>📷 Functions: grayscale, histogram equalize, blur, sharpen...</li>
            <li>⚙️ Built with OpenCV (Python)</li>
            <li>📚 Organized by chapter: 3, 4, 5, 9</li>
        </ul>
        """
    },
    {
        "icon": "🩺",
        "title": "Skin Disease Detection",
        "desc": """
        <ul>
            <li>🧠 Model: <code>ResNet18</code> được huấn luyện trên tập <code>HAM10000</code></li>
            <li>📸 Nhận ảnh qua webcam hoặc tải lên</li>
            <li>🔍 Phân loại 7 loại bệnh da liễu phổ biến</li>
            <li>📈 Dự đoán kèm xác suất và tên bệnh (Anh + Việt)</li>
        </ul>
        """
    }
]

    # Render
    for f in features:
        st.markdown(f"""
        <div class='feature-box'>
            <b>{f["icon"]} {f["title"]}</b><br>
            {f["desc"]}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📁 Related Files")
    st.markdown("""
    - `detect_asl.py`, `asl-detector.ipynb` – ASL detection (CNN + Mediapipe)
    - `predict.py` – Face recognition (ONNX + SVM)
    - `emotion.py` – Emotion detection
    - `fruit.py` – Fruit classification
    - `detect_obj.py`, `obj-detector.ipynb` – Object detection using YOLOv8
    - `skin_.py` – Skin disease detection
    - `requirements.txt` – Dependencies
    - `readme.md` – Project documentation
    """)
