import numpy as np
import cv2

def CreateMotionFilter(M, N, a=0.1, b=0.1, T=1.0):
    H = np.zeros((M, N), dtype=np.complex64)
    for u in range(M):
        for v in range(N):
            phi = np.pi * (a * (u - M // 2) + b * (v - N // 2))
            if abs(phi) < 1e-6:
                H[u, v] = T
            else:
                H[u, v] = T * (np.sin(phi) / phi) * np.exp(-1j * phi)
    return H

def CreateInverseMotionFilter(H, eps=1e-3):
    H_inv = np.zeros_like(H, dtype=np.complex64)
    mag = np.abs(H)
    H_inv[mag >= eps] = 1.0 / H[mag >= eps]
    return H_inv

def CreateWienerFilter(H, K=0.01):
    H_conj = np.conj(H)
    P = np.abs(H)**2
    G = H_conj / (P + K)
    return G

def ApplyFrequencyFilter(imgin, H_filter):
    f = imgin.astype(np.float64)
    F = np.fft.fft2(f)
    F_shift = np.fft.fftshift(F)
    G = F_shift * H_filter
    g = np.fft.ifft2(np.fft.ifftshift(G)).real
    return np.clip(g, 0, 255).astype(np.uint8)

def CreateMotion(imgin, a=0.1, b=0.1):
    M, N = imgin.shape
    H = CreateMotionFilter(M, N, a, b)
    return ApplyFrequencyFilter(imgin, H)

def DeMotion(imgin, a=0.1, b=0.1):
    M, N = imgin.shape
    H = CreateMotionFilter(M, N, a, b)
    H_inv = CreateInverseMotionFilter(H)
    return ApplyFrequencyFilter(imgin, H_inv)

def DeMotionWeiner(imgin, a=0.1, b=0.1, K=0.01):
    M, N = imgin.shape
    H = CreateMotionFilter(M, N, a, b)
    H_wiener = CreateWienerFilter(H, K)
    return ApplyFrequencyFilter(imgin, H_wiener)
