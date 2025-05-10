import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import os

# ==== Tên bệnh: Tiếng Anh + Tiếng Việt ====
label_names = [
    ("Melanocytic nevi (nv)", "Nốt ruồi lành tính"),
    ("Melanoma (mel)", "U hắc tố ác tính"),
    ("Benign keratosis-like lesions (bkl)", "Tổn thương sừng hóa lành tính"),
    ("Basal cell carcinoma (bcc)", "Ung thư biểu mô tế bào đáy"),
    ("Actinic keratoses (akiec)", "Dày sừng quang hóa"),
    ("Vascular lesions (vasc)", "Tổn thương mạch máu"),
    ("Dermatofibroma (df)", "U xơ da")
]

# ==== Load model từ thư mục hiện tại ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_resnet18.pth")

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ==== Hàm hiển thị giao diện ====
def show_skin():
    st.title("🩺 Phân loại bệnh da liễu (HAM10000)")
    uploaded_file = st.file_uploader("Tải ảnh da cần chẩn đoán", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(1).item()
            prob = torch.nn.functional.softmax(output, dim=1)[0][pred].item()

        eng, viet = label_names[pred]
        st.markdown(f"### 🧠 Kết quả: **{viet}**")
        st.caption(f"({eng})")
        st.markdown(f"🔍 Độ tin cậy: **{prob*100:.2f}%**")
