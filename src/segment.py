"""
Module tách ký tự từ ảnh biển số.

Tác giả: [Tên bạn]
Mô tả: Module này cung cấp các hàm để tách từng ký tự từ ảnh biển số 
       nhị phân đã được tiền xử lý.

Ghi chú cải tiến (theo phản hồi giáo viên):
- Chuyển RETR_EXTERNAL → RETR_TREE để tìm được contour bên trong khung viền biển số
- Thêm crop margin 5px để loại bỏ khung viền biển số trước khi tìm contour
- Lọc contour quá lớn (toàn bộ biển số hoặc khung viền)
- Lọc theo vị trí y (loại contour sát mép trên/dưới)
- Lọc contour lồng nhau (lỗ trong ký tự như 0, 8, A, B...)
- Tăng min_area lên 80 theo khuyến nghị của giáo viên
- Sửa default aspect_ratio_range: ký tự biển số VN có h/w ≈ 1.5–4.5
"""
import cv2
import numpy as np
import pandas as pd
import os


def segment_characters(binary_img, min_area=80, aspect_ratio_range=(1.4, 5.0),
                       row_threshold=10, margin=5, debug=False):
    """
    Tách từng ký tự từ ảnh biển số nhị phân.

    Giả định: chữ màu trắng (255) trên nền đen (0).

    Nguyên nhân 32 contour (theo phản hồi giáo viên) và cách khắc phục:
      1. RETR_EXTERNAL bị chặn bởi khung viền biển số → đổi sang RETR_TREE
      2. Khung viền trắng quanh biển được YOLO cắt rộng → crop thêm margin 5px
      3. Lọc contour quá lớn (chiếm > 80% chiều rộng hoặc chiều cao ảnh)
      4. Lọc contour sát mép trên/dưới (y < 10% hoặc y+h > 90% chiều cao ảnh)
      5. Lọc contour lồng nhau (lỗ bên trong ký tự như 8, 0, B, A...)
      6. Tăng min_area từ 40 lên 80 để loại bỏ chấm nhiễu nhỏ
      7. Sửa aspect_ratio_range: h/w của ký tự biển số VN thực tế ≈ 1.4–4.5

    Args:
        binary_img: Ảnh nhị phân (numpy.ndarray, 1 kênh).
        min_area (int): Diện tích bbox tối thiểu để coi là ký tự (mặc định 80).
        aspect_ratio_range (tuple): Khoảng tỉ lệ h/w hợp lệ (mặc định (1.4, 5.0)).
        row_threshold (int): Ngưỡng phân tách hàng ký tự (dựa trên y).
        margin (int): Số pixel crop thêm 4 cạnh để loại bỏ khung viền biển số.
        debug (bool): Nếu True, in ra số contour sau mỗi bước lọc.

    Returns:
        list: Danh sách các tuple (char_img, bbox) với bbox = (x, y, w, h),
              tọa độ đã tính theo ảnh gốc (trước crop margin).
    """
    # Đảm bảo chữ trắng (255), nền đen (0)
    if np.mean(binary_img) > 127:
        binary_img = 255 - binary_img

    h_img, w_img = binary_img.shape[:2]

    # ----------------------------------------------------------------
    # Cải tiến 6 (giáo viên): Crop thêm margin để loại bỏ khung viền
    # biển số (YOLO đôi khi cắt rộng hơn thực tế khiến threshold làm
    # nổi cả khung kim loại trắng xung quanh biển)
    # ----------------------------------------------------------------
    safe_margin = min(margin, h_img // 6, w_img // 6)
    if safe_margin > 0:
        cropped = binary_img[safe_margin:h_img - safe_margin,
                             safe_margin:w_img - safe_margin]
    else:
        cropped = binary_img
    h_crop, w_crop = cropped.shape[:2]

    # ----------------------------------------------------------------
    # Cải tiến 1 (giáo viên): Dùng RETR_TREE thay vì RETR_EXTERNAL
    #
    # RETR_EXTERNAL chỉ tìm contour ngoài cùng. Nếu biển số có khung
    # viền trắng kín thì RETR_EXTERNAL chỉ trả về khung đó, không tìm
    # thấy bất kỳ ký tự nào bên trong.
    # RETR_TREE tìm tất cả contour kể cả bên trong → phát hiện ký tự.
    # ----------------------------------------------------------------
    contours, hierarchy = cv2.findContours(
        cropped,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if debug:
        print(f"[DEBUG] Tổng contour tìm được: {len(contours)}")

    chars_raw = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = h / w if w > 0 else 0

        # ----------------------------------------------------------------
        # Cải tiến 3 (giáo viên): Lọc theo diện tích – loại bỏ chấm nhiễu
        # min_area = 80 theo khuyến nghị (trước đây là 40, còn nhiều nhiễu)
        # ----------------------------------------------------------------
        if area < min_area:
            continue

        # ----------------------------------------------------------------
        # Cải tiến khung viền lớn: loại contour chiếm > 80% chiều rộng/
        # chiều cao ảnh (khung viền biển hoặc toàn bộ biển số)
        # ----------------------------------------------------------------
        if w > 0.8 * w_crop or h > 0.8 * h_crop:
            continue

        # ----------------------------------------------------------------
        # Cải tiến 4 (giáo viên): Lọc theo tỉ lệ h/w
        # Ký tự biển số VN cao hơn rộng → h/w ≈ 1.4–4.5
        # ratio = w/h (giáo viên gọi là ratio = w/h)
        # → tương đương aspect = h/w phải trong (1.4, 5.0)
        # ----------------------------------------------------------------
        if aspect < aspect_ratio_range[0] or aspect > aspect_ratio_range[1]:
            continue

        # ----------------------------------------------------------------
        # Lọc thêm: chiều rộng tuyệt đối tối thiểu
        # Ký tự biển số VN có chiều rộng >= 10px (sau resize về h=180)
        # Loại bỏ sọc nhiễu mảnh dọc vẫn lọt qua aspect ratio filter
        # ----------------------------------------------------------------
        if w < 10:
            continue

        # ----------------------------------------------------------------
        # Cải tiến 5 (giáo viên): Lọc theo vị trí y – loại contour sát
        # mép trên và mép dưới biển số (thường là nhiễu từ viền kim loại)
        # ----------------------------------------------------------------
        if y < 0.05 * h_crop:
            continue
        if y + h > 0.95 * h_crop:
            continue

        # Khôi phục tọa độ về hệ ảnh gốc (trước crop margin)
        chars_raw.append((x + safe_margin, y + safe_margin, w, h, i))

    if debug:
        print(f"[DEBUG] Sau lọc area/aspect/y-boundary: {len(chars_raw)} contour")

    if not chars_raw:
        return []

    # ----------------------------------------------------------------
    # Cải tiến: Loại bỏ contour lồng nhau (lỗ bên trong ký tự như 8,
    # 0, B, A, D, Q...). Nếu bbox của contour A nằm hoàn toàn bên trong
    # bbox của contour B thì A là "lỗ" → loại bỏ.
    # ----------------------------------------------------------------
    valid_chars = []
    for idx1, c1 in enumerate(chars_raw):
        x1, y1, w1, h1, _ = c1
        is_nested = False
        for idx2, c2 in enumerate(chars_raw):
            if idx1 == idx2:
                continue
            x2, y2, w2, h2, _ = c2
            # c1 nằm hoàn toàn bên trong c2
            if (x1 >= x2 and y1 >= y2
                    and x1 + w1 <= x2 + w2
                    and y1 + h1 <= y2 + h2):
                is_nested = True
                break
        if not is_nested:
            valid_chars.append(c1)

    if debug:
        print(f"[DEBUG] Sau loại nested contour: {len(valid_chars)} contour")

    if not valid_chars:
        return []

    # Sắp xếp theo hàng (y) rồi cột (x)
    chars_sorted = sorted(valid_chars, key=lambda c: c[1])

    # Nhóm thành các hàng dựa trên ngưỡng row_threshold
    rows = []
    current_row = [chars_sorted[0]]
    for c in chars_sorted[1:]:
        if abs(c[1] - current_row[-1][1]) < row_threshold:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
    rows.append(current_row)

    # Sắp xếp từng hàng theo x và crop ký tự
    char_images = []
    for row in rows:
        row_sorted = sorted(row, key=lambda c: c[0])
        for (x, y, w, h, _) in row_sorted:
            char_crop = binary_img[y:y + h, x:x + w]
            char_images.append((char_crop, (x, y, w, h)))

    return char_images


def segment_with_contour(binary_img, min_area=80, aspect_ratio_range=(1.4, 5.0),
                         row_threshold=10, pad_char=2, margin=5):
    """
    Tách ký tự và thêm padding nhẹ cho mỗi ký tự.

    Args:
        binary_img: Ảnh nhị phân.
        min_area (int): Diện tích bbox tối thiểu (mặc định 80).
        aspect_ratio_range (tuple): Khoảng h/w hợp lệ (mặc định (1.4, 5.0)).
        row_threshold (int): Ngưỡng phân tách hàng.
        pad_char (int): Padding thêm cho mỗi ký tự (px).
        margin (int): Crop margin loại bỏ khung viền biển.

    Returns:
        list: Danh sách các tuple (char_img, bbox) đã có padding.
    """
    chars = segment_characters(
        binary_img, min_area, aspect_ratio_range, row_threshold, margin
    )

    if not chars:
        return []

    h_img, w_img = binary_img.shape[:2]
    padded_chars = []

    for char_img, (x, y, w, h) in chars:
        x1 = max(0, x - pad_char)
        y1 = max(0, y - pad_char)
        x2 = min(w_img, x + w + pad_char)
        y2 = min(h_img, y + h + pad_char)
        padded_img = binary_img[y1:y2, x1:x2]
        padded_chars.append((padded_img, (x1, y1, x2 - x1, y2 - y1)))

    return padded_chars


def evaluate_segmentation_on_folder(folder_path, ground_truth_counts=None, **seg_params):
    """
    Đánh giá segment trên một thư mục ảnh biển số đã tiền xử lý.

    Args:
        folder_path (str): Thư mục chứa ảnh nhị phân.
        ground_truth_counts (dict): Dict {filename: number_of_chars}.
        **seg_params: Các tham số cho segment_characters.

    Returns:
        dict: Kết quả đánh giá.
    """
    results = []

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        binary_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if binary_img is None:
            continue

        chars = segment_characters(binary_img, **seg_params)
        found = len(chars)

        gt = ground_truth_counts.get(fname, None) if ground_truth_counts else None

        results.append({
            'file': fname,
            'found': found,
            'ground_truth': gt,
            'correct': (found == gt) if gt is not None else None
        })

    df = pd.DataFrame(results)

    correct_rate = None
    if ground_truth_counts and not df.empty:
        mask = df['ground_truth'].notna()
        if mask.any():
            correct_rate = df[mask]['correct'].mean() * 100

    avg_found = df['found'].mean() if not df.empty else 0

    return {
        'details': df,
        'avg_found': avg_found,
        'correct_rate': correct_rate
    }


def grid_search_segmentation(folder_path, param_grid, ground_truth_counts=None):
    """
    Grid search cho tham số segmentation.

    Args:
        folder_path (str): Thư mục ảnh.
        param_grid (dict): Các tham số cần khảo sát.
        ground_truth_counts (dict): Ground truth số ký tự.

    Returns:
        pandas.DataFrame: Kết quả grid search.
    """
    results = []

    min_areas = param_grid.get('min_area', [80])
    aspect_ranges = param_grid.get('aspect_ratio_range', [(1.4, 5.0)])
    row_thresholds = param_grid.get('row_threshold', [10])

    total = len(min_areas) * len(aspect_ranges) * len(row_thresholds)
    print(f"Grid search segmentation với {total} tổ hợp tham số...")

    count = 0
    for ma in min_areas:
        for ar in aspect_ranges:
            for rt in row_thresholds:
                count += 1
                print(f"  [{count}/{total}] min_area={ma}, aspect={ar}, row={rt}")

                eval_res = evaluate_segmentation_on_folder(
                    folder_path,
                    ground_truth_counts,
                    min_area=ma,
                    aspect_ratio_range=ar,
                    row_threshold=rt
                )

                results.append({
                    'min_area': ma,
                    'aspect_min': ar[0],
                    'aspect_max': ar[1],
                    'row_threshold': rt,
                    'avg_found': eval_res['avg_found'],
                    'correct_rate': eval_res['correct_rate'] if eval_res['correct_rate'] is not None else 0
                })

    df = pd.DataFrame(results)
    return df.sort_values('correct_rate', ascending=False)