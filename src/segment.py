"""
Module tách ký tự từ ảnh biển số.

Tác giả: [Tên bạn]
Mô tả: Module này cung cấp các hàm để tách từng ký tự từ ảnh biển số 
       nhị phân đã được tiền xử lý.
"""
import cv2
import numpy as np
import pandas as pd
import os


def segment_characters(binary_img, min_area=40, aspect_ratio_range=(0.2, 1.6), 
                       row_threshold=10, debug=False):
    """
    Tách từng ký tự từ ảnh biển số nhị phân.
    
    Giả định: chữ màu trắng (255) trên nền đen (0).
    
    Args:
        binary_img (numpy.ndarray): Ảnh nhị phân.
        min_area (int): Diện tích tối thiểu của ký tự.
        aspect_ratio_range (tuple): Khoảng tỉ lệ h/w hợp lệ.
        row_threshold (int): Ngưỡng phân tách hàng (dựa trên y).
        debug (bool): Hiển thị ảnh contour để debug.
        
    Returns:
        list: Danh sách các tuple (char_img, bbox) với bbox = (x, y, w, h).
    """
    # Đảm bảo chữ trắng, nền đen
    if np.mean(binary_img) > 127:
        binary_img = 255 - binary_img
    
    # Tìm contour
    contours, hierarchy = cv2.findContours(
        binary_img, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if debug:
        # Vẽ contour lên ảnh
        img_contour = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', img_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = h / w if w > 0 else 0
        
        # Lọc theo diện tích và tỉ lệ
        if area < min_area:
            continue
        if aspect < aspect_ratio_range[0] or aspect > aspect_ratio_range[1]:
            continue
            
        # Lọc thêm: bỏ các contour ở rìa ảnh (nếu là viền)
        if x < 2 or y < 2 or x + w > binary_img.shape[1] - 2 or y + h > binary_img.shape[0] - 2:
            # Có thể là viền, nhưng vẫn giữ nếu không có lựa chọn khác
            pass
            
        chars.append((x, y, w, h, cnt))
    
    if not chars:
        return []
    
    # Sắp xếp theo hàng (y) và cột (x)
    chars_sorted = sorted(chars, key=lambda c: c[1])  # sort by y
    
    # Nhóm thành các hàng
    rows = []
    current_row = [chars_sorted[0]]
    for c in chars_sorted[1:]:
        if abs(c[1] - current_row[-1][1]) < row_threshold:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
    rows.append(current_row)
    
    # Sắp xếp từng hàng theo x và crop
    char_images = []
    for row in rows:
        row_sorted = sorted(row, key=lambda c: c[0])  # sort by x
        for (x, y, w, h, cnt) in row_sorted:
            char_crop = binary_img[y:y+h, x:x+w]
            char_images.append((char_crop, (x, y, w, h)))
    
    return char_images


def segment_with_contour(binary_img, min_area=40, aspect_ratio_range=(0.2, 1.6),
                         row_threshold=10, pad_char=2):
    """
    Tách ký tự và thêm padding nhẹ cho mỗi ký tự.
    
    Args:
        binary_img (numpy.ndarray): Ảnh nhị phân.
        min_area (int): Diện tích tối thiểu.
        aspect_ratio_range (tuple): Khoảng tỉ lệ h/w.
        row_threshold (int): Ngưỡng hàng.
        pad_char (int): Padding thêm cho mỗi ký tự (px).
        
    Returns:
        list: Danh sách các tuple (char_img, bbox) đã có padding.
    """
    chars = segment_characters(binary_img, min_area, aspect_ratio_range, row_threshold)
    
    if not chars:
        return []
    
    h_img, w_img = binary_img.shape[:2]
    padded_chars = []
    
    for char_img, (x, y, w, h) in chars:
        # Thêm padding
        x1 = max(0, x - pad_char)
        y1 = max(0, y - pad_char)
        x2 = min(w_img, x + w + pad_char)
        y2 = min(h_img, y + h + pad_char)
        padded_img = binary_img[y1:y2, x1:x2]
        padded_chars.append((padded_img, (x1, y1, x2-x1, y2-y1)))
    
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
    
    # Tính tỉ lệ đúng (nếu có ground truth)
    if gt is not None:
        correct_rate = df[df['ground_truth'].notna()]['correct'].mean() * 100
    else:
        correct_rate = None
    
    avg_found = df['found'].mean()
    
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
    
    min_areas = param_grid.get('min_area', [40])
    aspect_ranges = param_grid.get('aspect_ratio_range', [(0.2, 1.6)])
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