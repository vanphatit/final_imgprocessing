import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from spatial_transform import *
from frequency_filtering import *
from motion_blur_restore import *
from morphological_ops import *

def load_tif_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=2)
    return image

def plot_histogram(img, title):
    fig, ax = plt.subplots()
    ax.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
    ax.set_title(title)
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Tần suất")
    st.pyplot(fig)

def show():
    st.title("📚 DIP3E - Ứng dụng xử lý ảnh theo từng chương")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_dirs = {
        "Chương 3": os.path.join(BASE_DIR, "DIP3E_Original_Images_CH03"),
        "Chương 4": os.path.join(BASE_DIR, "DIP3E_Original_Images_CH04"),
        "Chương 5": os.path.join(BASE_DIR, "DIP3E_CH05_Original_Images"),
        "Chương 9": os.path.join(BASE_DIR, "DIP3E_Original_Images_CH09")
    }

    method_options = {
        "Chương 3": [
            "Negative", "Logarit", "Power", "Piecewise Linear",
            "Histogram", "HisEqual", "LocalHist",
            "Box Filter", "Gaussian Filter", "Median Filter", "Sharp"
        ],
        "Chương 4": ["Spectrum", "Remove Moire"],
        "Chương 5": ["Create Motion Blur", "DeMotion", "DeMotion Weiner"],
        "Chương 9": [
                        "Erosion", "Dilation", "Boundary",
                        "Contour", "Convex Hull", "Defect Detect",
                        "Hole Fill", "Connected Component", "Remove Small Rice"
                    ]
    }

    chapter_intros = {
        "Chương 3": "📌 **Biến đổi điểm và lọc không gian** — Âm bản, logarit, histogram, lọc trung bình, sharpening.",
        "Chương 4": "📌 **Biến đổi Fourier** — Phân tích phổ, loại bỏ nhiễu moiré trong miền tần số.",
        "Chương 5": "📌 **Khôi phục ảnh mờ** — Làm mờ chuyển động và phục hồi bằng Deconvolution/Wiener.",
        "Chương 9": "📌 **Hình thái học** — Erosion, Dilation, Boundary trích biên trên ảnh nhị phân."
    }

    method_desc = {
        "Negative": "📌 Biến đổi ảnh thành âm bản: `s = 255 - r`.",
        "Logarit": "📌 Tăng sáng vùng tối bằng hàm log: `s = c*log(1 + r)`.",
        "Power": "📌 Gamma correction: `s = c * r^γ`.",
        "Piecewise Linear": "📌 Biến đổi mức xám theo từng đoạn tuyến tính.",
        "Histogram": "📌 Hiển thị histogram ảnh gốc.",
        "HisEqual": "📌 Cân bằng histogram toàn cục.",
        "LocalHist": "📌 Cân bằng histogram cục bộ từng vùng.",
        "Box Filter": "📌 Lọc trung bình (mean filter).",
        "Gaussian Filter": "📌 Lọc Gaussian làm mịn ảnh.",
        "Median Filter": "📌 Lọc trung vị loại nhiễu muối tiêu.",
        "Sharp": "📌 Làm sắc nét ảnh bằng high-pass filter.",
        "Spectrum": "📌 Hiển thị phổ Fourier để phân tích tần số.",
        "Remove Moire": "📌 Khử nhiễu moiré bằng lọc tần số.",
        "Create Motion Blur": "📌 Làm mờ ảnh mô phỏng chuyển động.",
        "DeMotion": "📌 Khôi phục ảnh mờ bằng lọc nghịch đảo.",
        "DeMotion Weiner": "📌 Khôi phục ảnh mờ bằng lọc Wiener.",
        "Erosion": "📌 Co nhỏ đối tượng trắng — loại nhiễu nhỏ.",
        "Dilation": "📌 Nở rộng vùng trắng — lấp lỗ, nối nét.",
        "Boundary": "📌 Trích biên bằng `A - erosion(A)`.",
        "Contour": "📌 Tìm và vẽ contour (biên ngoài) của đối tượng trắng.",
        "Convex Hull": "📌 Vẽ đường bao lồi (convex hull) bao quanh đối tượng.",
        "Defect Detect": "📌 Phát hiện khuyết điểm hình dạng dựa trên convexity defects.",
        "Hole Fill": "📌 Tô lấp lỗ (hole filling) trong vùng trắng bằng flood fill.",
        "Connected Component": "📌 Đếm và đánh dấu các thành phần liên thông (connected components).",
        "Remove Small Rice": "📌 Xử lý ảnh để loại bỏ các hạt gạo nhỏ dựa trên morphology + connected components."
    }

    result_desc = {
        # CHƯƠNG 3
        "Negative": "Ảnh sau xử lý là ảnh âm bản, vùng sáng trở thành tối và ngược lại. Điều này giúp làm nổi bật các chi tiết bị ẩn trong vùng tối và hỗ trợ quan sát rõ hơn các cấu trúc có độ tương phản thấp.",
        "Logarit": "Biến đổi logarit làm sáng vùng tối và nén vùng sáng. Kết quả là ảnh rõ hơn ở phần thiếu sáng, thích hợp cho ảnh có dải cường độ hẹp ở mức thấp.",
        "Power": "Biến đổi gamma điều chỉnh độ sáng/độ tương phản tùy theo tham số γ. Gamma < 1 giúp sáng ảnh; gamma > 1 giúp tối ảnh. Hiệu quả trong việc làm nổi bật mức sáng cụ thể.",
        "Piecewise Linear": "Ảnh sau xử lý được tăng/giảm độ tương phản trong từng dải cường độ xác định trước. Phù hợp để nhấn mạnh vùng tối, trung tính hoặc sáng theo mục tiêu.",
        "Histogram": "Histogram mô tả phân bố mức xám trong ảnh gốc. Dạng phân bố cho biết ảnh có bị tối, sáng quá mức hoặc thiếu tương phản hay không.",
        "HisEqual": "Sau cân bằng histogram, độ tương phản ảnh được cải thiện rõ rệt. Ảnh sáng đều hơn, các chi tiết vùng tối hoặc sáng đều rõ hơn, nhưng có thể xuất hiện nhiễu nhẹ.",
        "LocalHist": "Cân bằng histogram cục bộ làm nổi bật chi tiết ở từng vùng. Ảnh kết quả có độ tương phản tốt hơn ở khu vực có điều kiện ánh sáng không đồng đều.",
        "Box Filter": "Lọc trung bình làm mượt ảnh bằng cách tính trung bình lân cận. Kết quả là ảnh mềm hơn nhưng có thể mất nét ở vùng biên hoặc chi tiết nhỏ.",
        "Gaussian Filter": "Ảnh sau xử lý được làm mượt tự nhiên nhờ trọng số Gaussian, ít mất biên hơn box filter. Thường dùng để tiền xử lý ảnh hoặc loại bỏ nhiễu nhẹ.",
        "Median Filter": "Lọc trung vị loại nhiễu muối tiêu hiệu quả, đặc biệt khi nhiễu không liên tục. Ảnh giữ được biên và chi tiết tốt hơn lọc trung bình.",
        "Sharp": "Lọc làm sắc nét giúp ảnh rõ hơn, đặc biệt là các cạnh và chi tiết nhỏ. Tuy nhiên, nếu dùng quá mức có thể gây nhiễu hoặc halo tại vùng biên.",

        # CHƯƠNG 4
        "Spectrum": "Hiển thị phổ Fourier giúp phân tích thành phần tần số trong ảnh. Các chi tiết lặp sẽ xuất hiện dưới dạng điểm sáng trong phổ, dùng để phát hiện nhiễu hoặc cấu trúc tuần hoàn.",
        "Remove Moire": "Lọc bỏ nhiễu moiré bằng cách làm mờ hoặc loại các thành phần tần số gây nhiễu. Kết quả là ảnh giảm các vân sóng hoặc nhiễu sọc, trở nên mượt và dễ quan sát hơn.",

        # CHƯƠNG 5
        "Create Motion Blur": "Ảnh sau xử lý mô phỏng hiệu ứng bị mờ do chuyển động tuyến tính (motion blur). Các chi tiết bị kéo dài theo hướng chuyển động đã định.",
        "DeMotion": "Lọc nghịch đảo giúp khôi phục chi tiết từ ảnh bị motion blur. Kết quả rõ hơn, nhưng dễ bị nhiễu hoặc viền giả nếu ảnh bị mờ nặng.",
        "DeMotion Weiner": "Lọc Wiener phục hồi ảnh mờ một cách tối ưu bằng cách giảm thiểu cả mờ và nhiễu. Kết quả mượt và ít nhiễu hơn lọc nghịch đảo, nhưng có thể mất một số chi tiết nếu ảnh gốc rất mờ.",

        # CHƯƠNG 9
        "Erosion": "Erosion làm co đối tượng trắng, loại bỏ điểm trắng nhỏ và nhiễu rìa. Kết quả ảnh mỏng đi ở vùng trắng, giúp tách vật thể sát nhau.",
        "Dilation": "Dilation làm phình vùng trắng, lấp các lỗ đen nhỏ và nối các thành phần gần nhau. Ảnh sau xử lý có đối tượng lớn hơn, liền mạch hơn.",
        "Boundary": "Trích biên bằng phép `A - erosion(A)` giúp xác định rìa đối tượng. Kết quả là ảnh chỉ giữ lại phần biên trắng, phần lõi bị loại bỏ, phù hợp để phát hiện hình dạng.",
        "Contour": "Ảnh sau xử lý hiển thị biên của đối tượng trắng, giúp xác định hình dạng và kiểm tra rìa vật thể.",
        "Convex Hull": "Đường bao lồi hiển thị khung bao ngoài chặt nhất quanh đối tượng, hữu ích trong nhận dạng hình học.",
        "Defect Detect": "Các điểm lõm được đánh dấu bằng hình tròn xanh, cho thấy khuyết điểm trên hình dạng so với bao lồi.",
        "Hole Fill": "Các vùng rỗng bên trong vật thể trắng được tô kín, giúp xử lý ảnh có lỗi về phân vùng.",
        "Connected Component": "Các thành phần liên thông được đếm và dán nhãn, hỗ trợ thống kê vật thể tách biệt.",
        "Remove Small Rice": "Ảnh kết quả loại bỏ các hạt gạo nhỏ không đạt ngưỡng diện tích, giữ lại hạt lớn nhất."
    }


    tabs = st.tabs(list(base_dirs.keys()))

    for i, chapter in enumerate(base_dirs):
        with tabs[i]:
            st.subheader(f"🗂 {chapter}")
            st.markdown(chapter_intros.get(chapter, ""), unsafe_allow_html=True)

            folder = base_dirs[chapter]
            try:
                image_files = [f for f in os.listdir(folder) if f.lower().endswith(".tif")]
            except FileNotFoundError:
                st.error(f"❌ Không tìm thấy thư mục `{folder}`.")
                continue

            colA, colB = st.columns([2, 2])

            with colA:
                selected_image = st.selectbox("🖼️ Chọn ảnh", image_files, key=f"{chapter}_img")

            with colB:
                choice = st.selectbox("🛠️ Xử lý", method_options[chapter], key=f"{chapter}_method")

            st.info(method_desc.get(choice, "📘 Không có mô tả."))

            img_path = os.path.join(folder, selected_image)
            img = load_tif_image(img_path)
            if img is None:
                st.warning("⚠️ Không thể đọc ảnh.")
                continue

            if img.ndim == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # === Xử lý ảnh ===
            out = None
            if chapter == "Chương 3":
                if choice == "Negative": out = Negative(gray)
                elif choice == "Logarit": out = Logarit(gray)
                elif choice == "Power": out = Power(gray)
                elif choice == "Piecewise Linear": out = PiecewiseLine(gray)
                elif choice == "Histogram": out = Histogram(gray)
                elif choice == "HisEqual": out = HisEqual(gray)
                elif choice == "LocalHist": out = LocalHist(gray)
                elif choice == "Box Filter": out = MySmoothBox(gray)
                elif choice == "Gaussian Filter": out = MySmoothGauss(gray)
                elif choice == "Median Filter": out = MyMedianFilter(gray)
                elif choice == "Sharp": out = Sharp(gray)
            elif chapter == "Chương 4":
                if choice == "Spectrum": out = Spectrum(gray)
                elif choice == "Remove Moire": out = RemoveMoireSimple(gray)
            elif chapter == "Chương 5":
                if choice == "Create Motion Blur": out = CreateMotion(gray)
                elif choice == "DeMotion": out = DeMotion(gray)
                elif choice == "DeMotion Weiner": out = DeMotionWeiner(gray)
            elif chapter == "Chương 9":
                if choice == "Erosion": out = Erosion(gray)
                elif choice == "Dilation": out = Dilation(gray)
                elif choice == "Boundary": out = Boundary(gray)
                elif choice == "Contour": out = Contour(gray)
                elif choice == "Convex Hull": out = ConvexHull(gray)
                elif choice == "Defect Detect": out = DefectDetect(gray)
                elif choice == "Hole Fill": out = HoleFill(gray)
                elif choice == "Connected Component": out = ConnectedComponent(gray)
                elif choice == "Remove Small Rice": out = RemoveSmallRice(gray)

            # === So sánh ảnh 2 cột ===
            st.markdown("### 🖼️ So sánh ảnh")
            col1, col2 = st.columns(2)

            with col1:
                st.image(img, caption=f"📥 Ảnh gốc: {selected_image}", use_container_width=True)

            with col2:
                if out is not None:
                    st.image(out, caption="📤 Kết quả xử lý", use_container_width=True)

            # === Histogram nếu là chương 3 ===
            if chapter == "Chương 3" and out is not None:
                st.markdown("### 📊 Histogram so sánh")
                colH1, colH2 = st.columns(2)
                with colH1: plot_histogram(gray, "Histogram ảnh gốc")
                with colH2: plot_histogram(out, "Histogram ảnh sau xử lý")

            # === Nhận xét ===
            st.markdown("### 📝 Nhận xét chi tiết")
            st.success(f"**Bạn đã thực hiện**: {method_desc.get(choice, '')}")
            st.info(result_desc.get(choice, "📘 Phân tích ảnh sau xử lý phụ thuộc vào kỹ thuật đã chọn."))

