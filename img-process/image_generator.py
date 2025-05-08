import numpy as np
import cv2

# === Dành cho chapter 3: spatial_transform ===
def create_gray_gradient(width=512, height=512):
    return np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))

def create_block_image(size=512):
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), 150, -1)
    cv2.rectangle(img, (300, 300), (400, 400), 255, -1)
    return img

def create_color_gradient(width=512, height=512):
    R = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    G = np.flipud(R)
    B = np.full((height, width), 128, dtype=np.uint8)
    return cv2.merge([B, G, R])

# === Dành cho chapter 4: frequency_filtering ===
def create_checkerboard(size=512, tile=64):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, tile*2):
        for j in range(0, size, tile*2):
            img[i:i+tile, j+tile:j+tile*2] = 255
            img[i+tile:i+tile*2, j:j+tile] = 255
    return img

# === Dành cho chapter 5: motion_blur_restore ===
def create_motion_blur_input(size=512):
    # Ảnh sáng giữa, tối biên
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (size//2, size//2), 100, 200, -1)
    return img

# === Dành cho chapter 9: morphological_ops ===
def create_binary_shape_image(size=512):
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (150, 150), 50, 255, -1)
    cv2.rectangle(img, (300, 300), (400, 400), 255, -1)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary
