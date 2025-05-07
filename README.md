# 🧠 AI Image Processing with Streamlit

Ứng dụng xử lý ảnh sử dụng các kỹ thuật AI hiện đại, giao diện thân thiện với người dùng qua Streamlit.

## 📂 Chức năng hiện có

- 👤 **Nhận diện khuôn mặt** (Realtime qua webcam)  
  Sử dụng model:
  - `YuNet` để phát hiện khuôn mặt (`.onnx`)
  - `SFace` để trích xuất embedding (`.onnx`)
  - `SVM` (`svc.pkl`) để phân loại danh tính

## 🚀 Cài đặt

### 1. Clone project

```bash
git clone https://github.com/yourusername/ai-image-processing.git
cd ai-image-processing
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

### 3. Cài các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## 🧪 Chạy ứng dụng

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại địa chỉ:

```
http://localhost:8501
```

## 📁 Cấu trúc thư mục

```
FINAL_IMGPROCESSING/
├── app.py                     # File Streamlit chính
├── requirements.txt
├── face-detect/              # Module nhận diện khuôn mặt
│   ├── predict.py
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   ├── svc.pkl
│   ├── stop.jpg
├── venv/                     # (Được bỏ qua trong .gitignore)
└── .gitignore
```

---

> Made by Văn Phát & Thanh Duy
