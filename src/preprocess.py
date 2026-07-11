"""
src/preprocess.py

Module tiền xử lý ảnh biển số cho pipeline ALPR.
Chỉ giữ lại các hàm cốt lõi phục vụ trực tiếp cho pipeline.
"""

from __future__ import annotations
import cv2
import numpy as np

def resize_if_small(img: np.ndarray, target_height: int = 180) -> tuple[np.ndarray, float]:
    """Phóng to ảnh crop nhỏ để ký tự rõ hơn."""
    h, w = img.shape[:2]

    if h >= target_height:
        return img.copy(), 1.0

    scale = target_height / max(h, 1)
    new_w = int(w * scale)

    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    return resized, float(scale)


def percentile_stretch(gray: np.ndarray, low: int = 2, high: int = 98) -> np.ndarray:
    """Kéo giãn tương phản ảnh xám để loại bỏ các điểm quá tối/sáng."""
    p_low, p_high = np.percentile(gray, (low, high))

    if p_high <= p_low:
        return gray.copy()

    out = (gray.astype(np.float32) - p_low) * 255.0 / (p_high - p_low)
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_gamma(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Tự động chỉnh sáng (gamma correction) cho ảnh tối hoặc quá sáng."""
    mean_value = gray.mean() / 255.0

    if mean_value < 0.35:
        gamma = 0.65
    elif mean_value > 0.75:
        gamma = 1.25
    else:
        gamma = 1.0

    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)])
    table = np.clip(table, 0, 255).astype(np.uint8)

    return cv2.LUT(gray, table), gamma


def unsharp_mask(gray: np.ndarray, amount: float = 1.1, sigma: float = 1.0) -> np.ndarray:
    """Làm nét chữ (dùng Gaussian Blur để unsharp mask)."""
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)

    return np.clip(sharp, 0, 255).astype(np.uint8)


def ensure_white_characters(binary: np.ndarray) -> np.ndarray:
    """Chuẩn hóa ảnh nhị phân để ký tự màu trắng, nền màu đen."""
    white_ratio = np.count_nonzero(binary == 255) / binary.size

    if white_ratio > 0.60:
        return cv2.bitwise_not(binary)

    return binary


def make_threshold_candidates(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Tạo các ảnh threshold ứng viên: Otsu và Adaptive."""
    candidates = {}

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates["otsu"] = ensure_white_characters(otsu)

    for block_size in [21, 31, 41]:
        if block_size < min(gray.shape[:2]):
            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                5,
            )
            candidates[f"adaptive{block_size}"] = ensure_white_characters(adaptive)

    return candidates


def choose_threshold(gray: np.ndarray, method: str = "auto") -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    """Chọn threshold (mặc định ưu tiên Otsu)."""
    candidates = make_threshold_candidates(gray)

    if method != "auto":
        if method not in candidates:
            raise ValueError(f"Không có threshold method={method}. Có thể dùng: {list(candidates)}")
        return method, candidates[method], candidates

    best_name = "otsu"
    return best_name, candidates[best_name], candidates


def apply_morphology(binary: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Làm sạch nhiễu nhỏ bằng morphology (Open -> Close)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed
