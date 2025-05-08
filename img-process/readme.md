Ứng dụng xử lý ảnh xây dựng bằng Python + Streamlit, sử dụng các hàm xử lý từ 4 chương lớn:

- `spatial_transform.py`: xử lý không gian & histogram
- `frequency_filtering.py`: xử lý ảnh trong miền tần số (DFT)
- `motion_blur_restore.py`: mô phỏng và phục hồi motion blur
- `morphological_ops.py`: phép toán hình thái học (Morphology)
- `image_generator.py`: tạo ảnh mẫu để test trực tiếp

---

- Có thể tải ảnh `.png`, `.jpg`, `.bmp`, `.tif` từ máy
- Hoặc chọn ảnh mẫu được tạo tự động phù hợp với từng nhóm chức năng
- Giao diện thông minh: khi chọn ảnh mẫu, chỉ hiện nhóm chức năng phù hợp để test

---

## 🧩 Nhóm chức năng & cách hoạt động

### 1. 🟠 Point Transforms (`spatial_transform.py`)

| Hàm | Mô tả |
|-----|------|
| `Negative()` | Đảo ảnh (âm bản): `s = L - 1 - r` |
| `Logarit()` | Tăng tương phản vùng tối: `s = c * log(1 + r)` |
| `Power()` | Biến đổi gamma: `s = c * r^γ` |
| `PiecewiseLine()` | Kéo giãn mức xám bằng đoạn thẳng ghép |

➡ Dùng để tăng cường ảnh, chỉnh độ sáng tương phản.

---

### 2. 🟡 Histogram Operations (`spatial_transform.py`)

| Hàm | Mô tả |
|-----|------|
| `Histogram()` | Vẽ histogram ảnh |
| `HisEqual()` | Cân bằng histogram toàn cục |
| `LocalHist()` | Cân bằng histogram cục bộ (3x3) |

➡ Dùng để cải thiện ảnh thiếu sáng, tương phản thấp.

---

### 3. 🔵 Filtering (`spatial_transform.py`)

| Hàm | Mô tả |
|-----|------|
| `MySmoothBox()` | Lọc trung bình |
| `MySmoothGauss()` | Lọc Gaussian |
| `MyMedianFilter()` | Lọc trung vị |
| `Sharp()` | Làm nét ảnh bằng Laplacian/LoG |

➡ Dùng để khử nhiễu hoặc làm rõ viền.

---

### 4. 🟣 Frequency Domain (`frequency_filtering.py`)

| Hàm | Mô tả |
|-----|------|
| `Spectrum()` | Tính phổ biên độ ảnh (DFT) |
| `RemoveMoireSimple()` | Lọc nhiễu Moiré bằng Notch Filter |

➡ Dùng để xử lý ảnh có nhiễu dạng sóng gợn định kỳ.

---

### 5. 🔴 Motion Blur (`motion_blur_restore.py`)

| Hàm | Mô tả |
|-----|------|
| `CreateMotion()` | Mô phỏng mờ do chuyển động |
| `DeMotion()` | Phục hồi ảnh bằng lọc nghịch đảo |
| `DeMotionWeiner()` | Phục hồi bằng lọc Weiner (chống nhiễu) |

➡ Dùng để xử lý ảnh bị mờ do rung lắc khi chụp.

---

### 6. 🟢 Morphology (`morphological_ops.py`)

| Hàm | Mô tả |
|-----|------|
| `Erosion()` | Làm mòn đối tượng (loại bỏ chi tiết nhỏ) |
| `Dilation()` | Phình to đối tượng |
| `Boundary()` | Trích đường biên bằng `Ảnh - Erosion(Ảnh)` |

➡ Dùng để tách biên, xử lý ảnh nhị phân.

---

## 🧪 Ảnh mẫu có sẵn (`image_generator.py`)

| Ảnh mẫu | Phù hợp test với |
|--------|------------------|
| Gray Gradient | Point Transforms, Histogram |
| Block Image | Filtering, Sharp, Histogram |
| Color Gradient | LogaritColor, HisEqualColor |
| Checkerboard | Frequency Domain |
| Motion Blur Circle | Motion Blur |
| Binary Shapes | Morphology |

---