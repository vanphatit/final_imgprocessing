# 🧠 AI Image Processing with Streamlit

An AI-powered image processing application with a user-friendly interface built using Streamlit.

---

## 🎬 Demo Video
[Watch on YouTube](https://youtu.be/NExNq7ONaGc)

---

## 📂 Features

- 👤 **Face Recognition (Realtime via Webcam)**
  - Face detection using `YuNet` (`.onnx`)
  - Embedding extraction with `SFace` (`.onnx`)
  - Identity classification using `SVM` (`svc.pkl`)

- 😊 **Emotion Detection**
  - CNN trained on FER2013
  - Classifies 7 basic emotions (happy, sad, angry, neutral, etc.)
  - Input via webcam or uploaded images

- 🤟 **ASL Hand Sign Recognition**
  - Based on MediaPipe landmarks
  - MobileNetV2 retrained on `dataset_retrain`
  - Supports A–Z (excluding J & Q)

- 🍎 **Fruit Classification**
  - Model: YOLOv8 (ONNX)
  - Dataset: Custom-collected fruit images
  - Detects durian, apple, dragon fruit, star fruit, soursop

- 📦 **Object Detection**
  - YOLOv8 pretrained on COCO dataset
  - Supports detection of 80 common object classes

- 🩺 **Skin Disease Diagnosis**
  - Model: ResNet18 trained on HAM10000
  - Classifies 7 common skin diseases (bilingual output: English & Vietnamese)

- 🖼️ **Traditional Image Processing (DIP3E Chapters 3, 4, 5, 9)**
  - Includes negative transform, log, histogram equalization, blur, sharpening, morphological operations, etc.

---

## 📁 Project Structure

```plaintext
app.py                      # Main Streamlit entry
home.py                     # Overview page
requirements.txt            # Python dependencies
README.md                   # This file
asl-detect/                 # ASL hand sign recognition
emotion-detect/             # Emotion detection
face-detect/                # Face recognition
fruit-detect/               # Fruit classification
obj-detect/                 # Object detection (YOLOv8)
skin-resnet18/              # Skin disease diagnosis
img-process/                # Traditional image processing
venv/                       # Virtual environment (excluded)
.gitignore
```

---

## 📊 Datasets Used

| Module                | Dataset        | Notes |
|-----------------------|----------------|-------|
| Emotion Detection     | FER2013        | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| Fruit Classification  | Custom         | [Manual labeling](https://drive.google.com/drive/folders/1ky5AdZl0mXE5CS_f2eUHzFq8ILz2KqT6?usp=sharing) |
| Skin Disease Diagnosis| HAM10000       | [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| ASL Recognition       | dataset_retrain| Collected via webcam |
| Object Detection      | COCO           | [COCO Dataset](https://cocodataset.org/#home) |

---

## ⚙️ Requirements

Install all required libraries via:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- streamlit
- opencv-python
- numpy
- tensorflow
- keras
- joblib
- ultralytics
- mediapipe
- matplotlib
- pillow
- scikit-learn

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-image-processing.git
cd ai-image-processing
```

2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate       # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Unzip dataset:
```bash
unzip asl-detect/dataset_retrain.zip -d asl-detect/dataset_retrain
```

5. Run the app:
```bash
streamlit run app.py
```

Then access: [http://localhost:8501](http://localhost:8501)

---

> Made with ❤️ by **Văn Phát & Thanh Duy**
