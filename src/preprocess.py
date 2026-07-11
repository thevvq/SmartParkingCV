"""
src/preprocess.py

Module tiền xử lý ảnh biển số cho pipeline ALPR.

Module này đã chỉnh để dùng được với cả 2 kiểu đầu vào:

1) Ảnh crop đã lưu thành file:
    result = preprocess_plate("dataset/crops/crop_01.jpg")

2) Ảnh crop dạng numpy array do YOLO trả về:
    crop_img, confidence = detector.crop_plate(image)
    result = preprocess_from_yolo_crop(crop_img)

Output chính đưa sang bước segment ký tự:
    binary_plate = result["morph"]

Vị trí trong pipeline:
    YOLO detect/crop biển số
    -> preprocess_plate(...) hoặc preprocess_from_yolo_crop(...)
    -> result["morph"]
    -> segment ký tự
    -> HOG/kNN/OCR
"""

from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. Các hàm xử lý ảnh cơ bản
# ============================================================

def resize_if_small(img: np.ndarray, target_height: int = 180) -> tuple[np.ndarray, float]:
    """
    Phóng to ảnh crop nhỏ để ký tự rõ hơn.

    Args:
        img: Ảnh BGR đầu vào.
        target_height: Chiều cao mục tiêu.

    Returns:
        resized: Ảnh sau resize.
        scale: Tỉ lệ phóng to.
    """
    h, w = img.shape[:2]

    if h >= target_height:
        return img.copy(), 1.0

    scale = target_height / max(h, 1)
    new_w = int(w * scale)

    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    return resized, float(scale)


def percentile_stretch(gray: np.ndarray, low: int = 2, high: int = 98) -> np.ndarray:
    """
    Kéo giãn tương phản ảnh xám.

    Dùng percentile để hạn chế ảnh hưởng của vài điểm quá tối/quá sáng.
    """
    p_low, p_high = np.percentile(gray, (low, high))

    if p_high <= p_low:
        return gray.copy()

    out = (gray.astype(np.float32) - p_low) * 255.0 / (p_high - p_low)
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_gamma(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Tự động chỉnh sáng cho ảnh tối hoặc quá sáng.

    gamma < 1: làm sáng ảnh.
    gamma > 1: làm tối ảnh quá sáng.
    """
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
    """Làm nét nét chữ biển số."""
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)

    return np.clip(sharp, 0, 255).astype(np.uint8)


# ============================================================
# 2. Threshold và Morphology
# ============================================================

def ensure_white_characters(binary: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa ảnh nhị phân để ký tự màu trắng, nền màu đen.

    Bước segment phía sau đang giả định chữ trắng trên nền đen.
    """
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


def score_binary(binary: np.ndarray) -> float:
    """
    Chấm điểm ảnh threshold.

    Điểm càng thấp càng tốt.
    Tiêu chí:
        - tỉ lệ pixel trắng hợp lý;
        - ít nhiễu nhỏ;
        - ít vùng trắng quá lớn.
    """
    white = np.count_nonzero(binary == 255) / binary.size

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])

    small_noise = np.sum(areas < 8)
    large_blobs = np.sum(areas > binary.size * 0.25)

    return abs(white - 0.28) * 3.0 + small_noise * 0.02 + large_blobs * 0.5


def choose_threshold(gray: np.ndarray, method: str = "auto") -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    """
    Chọn threshold Otsu/Adaptive hoặc tự động chọn ảnh tốt nhất.

    method:
        auto
        otsu
        adaptive21
        adaptive31
        adaptive41
    """
    candidates = make_threshold_candidates(gray)

    if method != "auto":
        if method not in candidates:
            raise ValueError(f"Không có threshold method={method}. Có thể dùng: {list(candidates)}")
        return method, candidates[method], candidates

    best_name = min(candidates, key=lambda name: score_binary(candidates[name]))
    return best_name, candidates[best_name], candidates


def remove_border_components(binary: np.ndarray) -> np.ndarray:
    """
    Xóa các vùng trắng lớn chạm mép ảnh.
    """
    h, w = binary.shape[:2]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = binary.copy()

    for label_id in range(1, num_labels):
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]

        touches_border = (
            x <= 1
            or y <= 1
            or x + bw >= w - 1
            or y + bh >= h - 1
        )

        looks_like_border_noise = (
            bw >= 0.45 * w
            or bh >= 0.70 * h
            or area >= 0.12 * h * w
        )

        if touches_border and looks_like_border_noise:
            cleaned[labels == label_id] = 0

    return cleaned

def clear_edge_strip(
        binary: np.ndarray,
        edge_ratio: float = 0.02,
    ) -> np.ndarray:
        """
        Tô đen một dải mỏng sát 4 mép ảnh.
        """
        h, w = binary.shape[:2]

        pad_y = max(1, int(h * edge_ratio))
        pad_x = max(1, int(w * edge_ratio))

        cleaned = binary.copy()

        cleaned[:pad_y, :] = 0
        cleaned[h - pad_y:, :] = 0
        cleaned[:, :pad_x] = 0
        cleaned[:, w - pad_x:] = 0

        return cleaned
def remove_plate_frame_lines(binary: np.ndarray) -> np.ndarray:
    """
    Xóa các đường khung ngang và dọc dài của biển số.

    Không xóa toàn bộ component nên ít làm mất ký tự hơn
    remove_border_components().
    """
    h, w = binary.shape[:2]

    horizontal_length = max(15, int(w * 0.35))
    vertical_length = max(15, int(h * 0.55))

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length),
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    frame_lines = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines,
    )

    cleaned = cv2.bitwise_and(
        binary,
        cv2.bitwise_not(frame_lines),
    )

    return cleaned
def apply_morphology(
    binary: np.ndarray,
    kernel_size: int = 2,
) -> np.ndarray:
    """
    Làm sạch nhiễu nhỏ bằng morphology nhẹ.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )

    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return closed


# ============================================================
# 3. Metric đánh giá nhỏ cho phần tiền xử lý
# ============================================================

def contrast_score(gray: np.ndarray) -> float:
    """Độ tương phản đơn giản: độ lệch chuẩn mức xám."""
    return float(np.std(gray))


def estimate_character_components(binary: np.ndarray) -> int:
    """
    Ước lượng số component có hình dáng gần giống ký tự.

    Đây không phải độ chính xác nhận diện cuối cùng.
    Nó chỉ là chỉ số nhỏ để kiểm tra chất lượng tiền xử lý.
    """
    h, w = binary.shape[:2]
    area = h * w

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    count = 0

    for i in range(1, num_labels):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        component_area = stats[i, cv2.CC_STAT_AREA]
        aspect = bw / max(bh, 1)

        valid_area = max(8, int(0.0005 * area)) <= component_area <= int(0.18 * area)
        valid_shape = 0.05 <= aspect <= 1.4 and bh >= 0.18 * h

        if valid_area and valid_shape:
            count += 1

    return count


# ============================================================
# 4. Hàm lõi: xử lý ảnh crop dạng numpy array
# ============================================================

def _preprocess_image(
    img: np.ndarray,
    source_name: str = "yolo_crop",
    save_path: str | Path | None = None,
    target_height: int = 180,
    clip_limit: float = 2.0,
    threshold_method: str = "auto",
    morph_kernel: int = 2,
) -> dict:
    """
    Hàm lõi xử lý ảnh crop biển số.

    Hàm này nhận ảnh dạng numpy array.
    preprocess_plate() và preprocess_from_yolo_crop() đều gọi vào hàm này.
    """
    if img is None:
        raise ValueError("Ảnh đầu vào không được là None")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    original = img.copy()
    h, w = original.shape[:2]

    margin_y = int(h * 0.05)
    margin_x = int(w * 0.03)

    original = original[
        margin_y:h-margin_y,
        margin_x:w-margin_x
]

    resized, scale = resize_if_small(original, target_height=target_height)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    stretched = percentile_stretch(gray)
    gamma_img, gamma_value = auto_gamma(stretched)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gamma_img)

    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=45, sigmaSpace=45)
    sharp = unsharp_mask(denoised)

    chosen_threshold, binary, candidates = choose_threshold(denoised, method=threshold_method)
    binary = clear_edge_strip(
    binary,
    edge_ratio=0.02,
)
    binary = remove_plate_frame_lines(binary)
    morph = apply_morphology(binary, kernel_size=morph_kernel)

    saved_to = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), morph)
        saved_to = str(save_path)

    return {
        "crop_path": source_name,
        "original": original,
        "resized": resized,
        "gray": gray,
        "stretched": stretched,
        "gamma": gamma_img,
        "enhanced": enhanced,
        "denoised": denoised,
        "sharp": sharp,
        "binary": binary,
        "morph": morph,
        "threshold_candidates": candidates,
        "threshold_method": chosen_threshold,
        "scale": scale,
        "gamma_value": gamma_value,
        "preprocessed_path": saved_to,
        "metrics": {
            "contrast_before": contrast_score(gray),
            "contrast_after": contrast_score(sharp),
            "white_ratio": float(np.count_nonzero(morph) / morph.size),
            "estimated_char_components": estimate_character_components(morph),
        },
    }


# ============================================================
# 5. Hàm chính cho pipeline gọi
# ============================================================

def preprocess_plate(
    crop_path: str | Path,
    save_path: str | Path | None = None,
    target_height: int = 180,
    clip_limit: float = 2.0,
    threshold_method: str = "auto",
    morph_kernel: int = 2,
) -> dict:
    """
    Tiền xử lý ảnh biển số từ đường dẫn file.

    Dùng khi:
        - test ảnh trong dataset/crops;
        - YOLO đã lưu crop ra file;
        - pipeline có sẵn crop_path.

    Output chính:
        result["morph"]
    """
    crop_path = Path(crop_path)
    img = cv2.imread(str(crop_path))

    if img is None:
        raise ValueError(f"Không đọc được ảnh: {crop_path}")

    return _preprocess_image(
        img=img,
        source_name=str(crop_path),
        save_path=save_path,
        target_height=target_height,
        clip_limit=clip_limit,
        threshold_method=threshold_method,
        morph_kernel=morph_kernel,
    )


def preprocess_from_yolo_crop(
    crop_img: np.ndarray,
    save_path: str | Path | None = None,
    source_name: str = "yolo_crop",
    target_height: int = 180,
    clip_limit: float = 2.0,
    threshold_method: str = "auto",
    morph_kernel: int = 2,
) -> dict:
    """
    Tiền xử lý trực tiếp ảnh crop do YOLO trả về.

    Dùng khi YOLO đang trả về:
        crop_img, confidence = detector.crop_plate(image)

    Khi đó pipeline có thể gọi:
        preprocess_result = preprocess_from_yolo_crop(crop_img)
        binary_plate = preprocess_result["morph"]
    """
    return _preprocess_image(
        img=crop_img,
        source_name=source_name,
        save_path=save_path,
        target_height=target_height,
        clip_limit=clip_limit,
        threshold_method=threshold_method,
        morph_kernel=morph_kernel,
    )


# Tên phụ để notebook cũ vẫn gọi được.
preprocess_plate_v2 = preprocess_plate


# ============================================================
# 6. Lưu ảnh debug hàng ngang
# ============================================================

def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh xám sang BGR để ghép hình."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.copy()


def _resize_for_debug(image: np.ndarray, width: int = 180, height: int = 120) -> np.ndarray:
    """Resize ảnh về cùng kích thước để ghép hàng ngang dễ nhìn."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _add_title_above(image: np.ndarray, title: str, title_height: int = 28) -> np.ndarray:
    """Thêm tiêu đề ở phía trên ảnh, không che lên ảnh."""
    image = _to_bgr(image)
    title_bar = np.full((title_height, image.shape[1], 3), 255, dtype=np.uint8)

    cv2.putText(
        title_bar,
        title,
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([title_bar, image])


def save_debug_grid_horizontal(result: dict, save_path: str | Path) -> str:
    """
    Lưu ảnh minh họa các bước tiền xử lý theo hàng ngang.

    Hàng ngang:
        original | resized | gray | stretched | gamma
        | clahe | bilateral | sharp | binary | morph
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    steps = [
        ("original", result["original"]),
        ("resized", result["resized"]),
        ("gray", result["gray"]),
        ("stretched", result["stretched"]),
        ("gamma", result["gamma"]),
        ("clahe", result["enhanced"]),
        ("bilateral", result["denoised"]),
        ("sharp", result["sharp"]),
        ("binary", result["binary"]),
        ("morph", result["morph"]),
    ]

    panels = []
    for title, image in steps:
        image_bgr = _to_bgr(image)
        image_bgr = _resize_for_debug(image_bgr, width=180, height=120)
        panel = _add_title_above(image_bgr, title)
        panels.append(panel)

    grid = np.hstack(panels)
    cv2.imwrite(str(save_path), grid)

    return str(save_path)


# Giữ tên cũ nếu pipeline/notebook đang gọi save_debug_grid.
save_debug_grid = save_debug_grid_horizontal


def save_preprocessing_steps(result: dict, output_dir: str | Path) -> None:
    """Lưu từng ảnh trung gian thành từng file riêng."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_name = Path(result["crop_path"]).stem

    steps = {
        "01_original": result["original"],
        "02_resized": result["resized"],
        "03_gray": result["gray"],
        "04_stretched": result["stretched"],
        "05_gamma": result["gamma"],
        "06_clahe": result["enhanced"],
        "07_denoised": result["denoised"],
        "08_sharp": result["sharp"],
        "09_binary": result["binary"],
        "10_morph": result["morph"],
    }

    for step_name, image in steps.items():
        save_path = output_dir / f"{crop_name}_{step_name}.png"
        cv2.imwrite(str(save_path), image)


# ============================================================
# 7. Chạy nhiều ảnh trong folder
# ============================================================

def preprocess_directory(
    input_dir: str | Path = "dataset/crops",
    output_dir: str | Path = "outputs/preprocessed",
    summary_csv: str | Path = "outputs/preprocessing_summary.csv",
    debug_dir: str | Path | None = "outputs/debug_grids",
    save_steps: bool = False,
) -> pd.DataFrame:
    """Chạy tiền xử lý cho toàn bộ ảnh crop trong thư mục."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    summary_csv = Path(summary_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg"))
        + list(input_dir.glob("*.png"))
    )

    if not image_files:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong {input_dir}")

    rows = []

    for image_path in image_files:
        save_path = output_dir / f"{image_path.stem}_preprocessed.png"

        result = preprocess_plate(
    crop_path=image_path,
    save_path=save_path,
    threshold_method="otsu",
)

        debug_grid_path = None
        if debug_dir is not None:
            debug_grid_path = Path(debug_dir) / f"{image_path.stem}_debug_grid.jpg"
            save_debug_grid_horizontal(result, debug_grid_path)

        if save_steps:
            steps_dir = output_dir.parent / "preprocessing_steps" / image_path.stem
            save_preprocessing_steps(result, steps_dir)

        rows.append({
            "image": image_path.name,
            "preprocessed_path": result["preprocessed_path"],
            "debug_grid_path": str(debug_grid_path) if debug_grid_path else None,
            "threshold_method": result["threshold_method"],
            "gamma_value": round(result["gamma_value"], 3),
            "scale": round(result["scale"], 3),
            "contrast_before": round(result["metrics"]["contrast_before"], 3),
            "contrast_after": round(result["metrics"]["contrast_after"], 3),
            "white_ratio": round(result["metrics"]["white_ratio"], 4),
            "estimated_char_components": result["metrics"]["estimated_char_components"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(summary_csv, index=False)

    return df


def preprocess_from_yolo_result(
    yolo_result: list[dict],
    output_dir: str | Path = "outputs/preprocessed",
    debug_dir: str | Path | None = "outputs/debug_grids",
) -> list[dict]:
    """
    Nhận kết quả từ YOLO rồi tiền xử lý từng crop.

    Hỗ trợ 2 kiểu item:

    Kiểu 1: YOLO đã lưu crop ra file
        {
            "crop_path": "outputs/yolo_crops/car_01_crop.jpg",
            "confidence_detection": 0.91,
            "bbox": [x1, y1, x2, y2]
        }

    Kiểu 2: YOLO trả crop trực tiếp dạng numpy array
        {
            "crop_img": crop_img,
            "confidence_detection": 0.91,
            "bbox": [x1, y1, x2, y2]
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    items = []

    for idx, item in enumerate(yolo_result):
        confidence = item.get("confidence_detection", item.get("confidence", None))
        bbox = item.get("bbox", None)

        if "crop_path" in item:
            crop_path = Path(item["crop_path"])
            save_path = output_dir / f"{crop_path.stem}_preprocessed.png"

            result = preprocess_plate(
                crop_path=crop_path,
                save_path=save_path,
            )
            crop_name = crop_path.stem

        elif "crop_img" in item:
            crop_name = f"yolo_crop_{idx:03d}"
            save_path = output_dir / f"{crop_name}_preprocessed.png"

            result = preprocess_from_yolo_crop(
                crop_img=item["crop_img"],
                save_path=save_path,
                source_name=crop_name,
            )

        elif "cropped_image" in item:
            crop_name = f"yolo_crop_{idx:03d}"
            save_path = output_dir / f"{crop_name}_preprocessed.png"

            result = preprocess_from_yolo_crop(
                crop_img=item["cropped_image"],
                save_path=save_path,
                source_name=crop_name,
            )

        else:
            raise ValueError("YOLO item cần có 'crop_path', 'crop_img' hoặc 'cropped_image'.")

        debug_grid_path = None
        if debug_dir is not None:
            debug_grid_path = Path(debug_dir) / f"{crop_name}_debug_grid.jpg"
            save_debug_grid_horizontal(result, debug_grid_path)

        items.append({
            "crop_name": crop_name,
            "preprocessed_path": result["preprocessed_path"],
            "debug_grid_path": str(debug_grid_path) if debug_grid_path else None,
            "threshold_method": result["threshold_method"],
            "scale": result["scale"],
            "gamma_value": result["gamma_value"],
            "estimated_char_components": result["metrics"]["estimated_char_components"],
            "confidence_detection": confidence,
            "bbox": bbox,
            "morph": result["morph"],
        })

    return items


# ============================================================
# 8. Chạy bằng Terminal
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý ảnh crop biển số.")
    parser.add_argument("--input",  default="valid/images", help="Thư mục ảnh crop biển số.")
    parser.add_argument("--output", default="outputs/preprocessed", help="Thư mục lưu ảnh cuối sau tiền xử lý.")
    parser.add_argument("--summary", default="outputs/preprocessing_summary.csv", help="File CSV tổng hợp.")
    parser.add_argument("--debug-dir", default="outputs/debug_grids", help="Thư mục lưu ảnh debug hàng ngang.")
    parser.add_argument("--save-steps", action="store_true", help="Lưu từng ảnh trung gian thành từng file riêng.")

    args = parser.parse_args()

    df = preprocess_directory(
        input_dir=args.input,
        output_dir=args.output,
        summary_csv=args.summary,
        debug_dir=args.debug_dir,
        save_steps=args.save_steps,
    )

    print("Đã chạy xong tiền xử lý.")
    print(f"Số ảnh: {len(df)}")
    print(f"Output ảnh cuối: {args.output}")
    print(f"Debug grid hàng ngang: {args.debug_dir}")
    print(f"Summary: {args.summary}")
    print(df)


if __name__ == "__main__":
    main()
