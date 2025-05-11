
# 🧠 AI Image Processing with Streamlit

Ứng dụng xử lý ảnh sử dụng các kỹ thuật AI hiện đại, giao diện thân thiện với người dùng qua Streamlit.

---

## Video demo:
https://youtu.be/NExNq7ONaGc

---

## 📂 Chức năng hiện có

- 👤 **Nhận diện khuôn mặt** (Realtime qua webcam)  
  Sử dụng model:
  - `YuNet` để phát hiện khuôn mặt (`.onnx`)
  - `SFace` để trích xuất embedding (`.onnx`)
  - `SVM` (`svc.pkl`) để phân loại danh tính

- 😊 **Nhận diện cảm xúc** (CNN - FER2013)  
  - Phân loại 7 cảm xúc cơ bản (happy, sad, angry, neutral, v.v.)
  - Nhập qua webcam hoặc ảnh tải lên

- 🤟 **Nhận diện ký hiệu tay ASL**  
  - Dựa trên landmark từ MediaPipe  
  - Mô hình MobileNetV2 huấn luyện lại từ tập `dataset_retrain`  
  - Realtime nhận diện A-Z (trừ J & Q)

- 🍎 **Phân loại trái cây**  
  - Model: YOLOv8 ONNX  
  - Dataset: Ảnh trái cây do nhóm tự thu thập  
  - Nhận diện sầu riêng, táo, thanh long, khế, mãng cầu xiêm.

- 📦 **Nhận diện đối tượng**  
  - YOLOv8 (PyTorch) pretrained trên COCO  
  - Phát hiện 80 lớp đối tượng từ ảnh tĩnh

- 🩺 **Chẩn đoán bệnh da liễu**  
  - Model: ResNet18 huấn luyện trên HAM10000  
  - Phân loại 7 loại bệnh da thường gặp (song ngữ Anh – Việt)

- 🖼️ **Xử lý ảnh truyền thống** (DIP3E Chương 3, 4, 5, 9)  
  - Âm bản, logarit, histogram, lọc mờ, sharpening, morphology...

---

## 📁 Project Structure

├── app.py                        # Trang chủ Streamlit  
├── home.py                       # Trang giới thiệu tổng quan  
├── requirements.txt              # Thư viện cần thiết  
├── README.md                     # Tài liệu dự án  
├── hcmute.png                    # Logo  
├── van-phat.jpg / thanh-duy.jpg # Ảnh thành viên nhóm  

├── asl-detect/                   # Nhận diện ký hiệu tay ASL  
│   ├── detect_asl.py  
│   ├── collect-data-retain.py  
│   ├── asl-detector-retrained-*.h5  # Các mô hình huấn luyện lại  
│   ├── class_names.txt  
│   ├── dataset_retrain.zip  
│   └── ding.mp3                  # Âm thanh phản hồi  

├── emotion-detect/              # Nhận diện cảm xúc  
│   ├── emotion.py  
│   ├── emotion_model.h5  
│   └── emotion-detect.ipynb  

├── face-detect/                 # Nhận diện khuôn mặt  
│   ├── predict.py  
│   ├── face_detection_yunet_*.onnx  
│   ├── face_recognition_sface_*.onnx  
│   └── svc.pkl  

├── fruit-detect/                # Nhận diện trái cây  
│   └── fruit.py  

├── obj-detect/                  # Nhận diện đối tượng YOLOv8  
│   ├── detect_obj.py  
│   ├── obj-detector.ipynb  
│   ├── yolov8_coco_trained.pt  
│   └── test-image.jpg  

├── skin-resnet18/               # Chẩn đoán bệnh da liễu  
│   ├── skin_resnet18.py  
│   ├── skin_resnet18.pth  
│   └── skin-resnet18.ipynb  

├── img-process/                 # Xử lý ảnh truyền thống (DIP3E)  
│   ├── DIP3E_Original_Images_CH03/  
│   ├── DIP3E_Original_Images_CH04/  
│   ├── DIP3E_CH05_Original_Images/  
│   ├── DIP3E_Original_Images_CH09/  
│   ├── processing.py  
│   ├── spatial_transform.py  
│   ├── frequency_filtering.py  
│   ├── motion_blur_restore.py  
│   ├── morphological_ops.py  
│   └── image_generator.py  

├── venv/                        # Môi trường ảo (bỏ qua trong Git)
└── .gitignore

---

## 📊 Datasets

| Module                | Dataset          | Ghi chú                                      |
|-----------------------|------------------|----------------------------------------------|
| Emotion Detection     | FER2013          | [Public dataset](https://www.kaggle.com/datasets/msambare/fer2013) |
| Fruit Classification  | Tự thu thập      | [Ảnh từ Internet, tự gán nhãn thủ công](https://drive.google.com/drive/folders/1ky5AdZl0mXE5CS_f2eUHzFq8ILz2KqT6?usp=sharing)       |
| Skin Disease Detection| HAM10000         | [Dataset bệnh da - Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| ASL Sign Language     | dataset_retrain  | Tự gán nhãn từ webcam                        |
| Object Detection      | COCO             | [Common Objects in Context](https://cocodataset.org/#home) |

---

## ⚙️ Yêu cầu thư viện

streamlit  
opencv-python  
numpy  
tensorflow  
keras  
joblib  
ultralytics  
mediapipe  
matplotlib  
pillow  
scikit-learn

> Cài nhanh bằng:

```
pip install -r requirements.txt
```

---

## 🚀 Cài đặt

### 1. Clone project

```
git clone https://github.com/yourusername/ai-image-processing.git
cd ai-image-processing
```

### 2. Tạo môi trường ảo (khuyến nghị)

```
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

### 3. Cài các thư viện cần thiết

```
pip install -r requirements.txt
```

### 4. Giải nén ../asl-detect/dataset_retrain.zip
```
unzip ../asl-detect/dataset_retrain.zip -d ../asl-detect/dataset_retrain
```

---

## 🧪 Chạy ứng dụng

```
streamlit run app.py
```

Truy cập tại:  
👉 http://localhost:8501

---

> Made with ❤️ by **Văn Phát & Thanh Duy**
