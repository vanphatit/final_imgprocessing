import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import os

# ==== Nhãn bệnh: (tiếng Anh, tiếng Việt) ====
label_names = [
    ("Melanocytic nevi (nv)", "Nốt ruồi lành tính"),
    ("Melanoma (mel)", "U hắc tố ác tính"),
    ("Benign keratosis-like lesions (bkl)", "Tổn thương sừng hóa lành tính"),
    ("Basal cell carcinoma (bcc)", "Ung thư biểu mô tế bào đáy"),
    ("Actinic keratoses (akiec)", "Dày sừng quang hóa"),
    ("Vascular lesions (vasc)", "Tổn thương mạch máu"),
    ("Dermatofibroma (df)", "U xơ da")
]

# ==== Load model ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_resnet18.pth")

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(label_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ==== Giao diện chính ====
def show_skin():
    st.title("🩺 Phân loại Bệnh Da Liễu - HAM10000")
    st.markdown("Mô hình học sâu **ResNet18** được huấn luyện để phân loại 7 loại bệnh da thường gặp.")

    # === Upload ảnh ===
    uploaded_file = st.file_uploader("📤 Vui lòng tải lên ảnh vùng da cần chẩn đoán", type=["jpg", "jpeg", "png"])

    # === Thông tin mô hình ===
    with st.expander("📊 Thông tin mô hình", expanded=False):
        st.markdown(f"- **Tên mô hình**: `skin_resnet18.pth`")
        st.markdown(f"- **Số lớp phân loại**: `{len(label_names)}`")
        st.markdown("### 📚 Các lớp có thể nhận diện:")
        cols = st.columns(2)
        for i, (eng, viet) in enumerate(label_names):
            with cols[i % 2]:
                st.markdown(f"- **{viet}**  \n  _({eng})_")

    # === Dự đoán nếu có ảnh ===
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Ảnh đã tải lên", use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(1).item()
            prob = torch.nn.functional.softmax(output, dim=1)[0][pred].item()

        eng, viet = label_names[pred]
        st.success(f"✅ **Kết quả**: {viet}")
        st.caption(f"_({eng})_")
        st.markdown(f"📈 **Độ tin cậy**: `{prob*100:.2f}%`")
    else:
        st.info("🖼️ Hãy tải lên một ảnh để bắt đầu chẩn đoán.")

