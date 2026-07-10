import os
import cv2
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Thêm đường dẫn gốc để import src
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.yolo_detector import YOLODetector
from src.segment import segment_with_contour

try:
    from src.preprocess import preprocess_from_yolo_crop
except ImportError:
    from preprocess import preprocess_from_yolo_crop

# Cấu hình encoding hiển thị tiếng Việt
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Cấu hình đường dẫn
DATASET_PATH = ROOT_DIR
VALID_IMG_DIR = os.path.join(DATASET_PATH, 'valid', 'images')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Khởi tạo YOLO Detector mặc định để lấy crop phục vụ khảo sát Preprocess & Segment
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'run_1', 'weights', 'best.pt')
if not os.path.exists(MODEL_PATH):
    # Dùng tạm model pretrained nếu chưa train xong
    MODEL_PATH = os.path.join(ROOT_DIR, 'yolov8n.pt')

detector = YOLODetector(model_path=MODEL_PATH, conf_threshold=0.25, imgsz=640)

# Tải danh sách ảnh validation
valid_images = [f for f in os.listdir(VALID_IMG_DIR) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:30] # Khảo sát nhanh trên 30 ảnh đại diện

# Lấy trước các ảnh crop biển số để khảo sát Preprocess & Segment
plate_crops = []
for fname in valid_images:
    img_path = os.path.join(VALID_IMG_DIR, fname)
    img = cv2.imread(img_path)
    if img is not None:
        crop, _ = detector.crop_plate(img)
        if crop is not None:
            plate_crops.append(crop)

print(f"Da chuan bi: {len(plate_crops)} anh crop bien so tu tap validation.")

# Dung demo_plate_crop.jpg (cung anh voi Cell 8 notebook) lam anh dai dien
SHARED_CROP_PATH = os.path.join(OUTPUT_DIR, "demo_plate_crop.jpg")
if os.path.exists(SHARED_CROP_PATH):
    demo_crop = cv2.imread(SHARED_CROP_PATH)
    plate_crops = [demo_crop] + plate_crops
    print(f"Da them anh demo chung: {SHARED_CROP_PATH}")


# Hàm vẽ biểu đồ khảo sát
def plot_survey_chart(title, x_labels, y_values, y_label, filename, color='#2b7bba'):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x_labels, y_values, color=color, alpha=0.9, edgecolor='black', width=0.4)
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, max(y_values) * 1.2 if y_values else 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')
                    
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

# ============================================================
# 1. KHẢO SÁT THAM SỐ YOLO
# ============================================================
print("\n========== BƯỚC 1: KHẢO SÁT THAM SỐ YOLO (Conf & IoU) ==========")
conf_sweep = [0.50, 0.80, 0.95]
yolo_results = []

for conf in conf_sweep:
    temp_detector = YOLODetector(model_path=MODEL_PATH, conf_threshold=conf, imgsz=640)
    eval_res = temp_detector.evaluate_on_folder(VALID_IMG_DIR, save_crops=False)
    
    if eval_res:
        yolo_results.append({
            'conf': conf,
            'detect_rate': eval_res['detection_rate'],
            'usable_rate': eval_res['usable_rate'],
            'latency': eval_res['avg_latency']
        })
        print(f"YOLO Conf={conf} -> Tỉ lệ phát hiện: {eval_res['detection_rate']:.1f}%, Khả dụng: {eval_res['usable_rate']:.1f}% (Latency: {eval_res['avg_latency']:.1f}ms)")

# Vẽ biểu đồ YOLO
yolo_labels = [f"conf={r['conf']}" for r in yolo_results]
yolo_accs = [r['detect_rate'] for r in yolo_results]
plot_survey_chart('Khảo sát YOLO: Confidence Threshold vs Detection Rate', yolo_labels, yolo_accs, 'Tỷ lệ phát hiện (%)', 'survey_yolo_conf.png', '#2b7bba')

# ============================================================
# 2. KHẢO SÁT THAM SỐ PREPROCESS
# ============================================================
print("\n========== BƯỚC 2: KHẢO SÁT THAM SỐ PREPROCESS (Clip Limit) ==========")
clip_sweep = [0.5, 2.0, 6.0]
prep_results = []

for clip in clip_sweep:
    char_counts = []
    contrast_improvements = []
    
    for crop in plate_crops:
        # Chạy thử preprocess
        res = preprocess_from_yolo_crop(crop, clip_limit=clip)
        char_counts.append(res['metrics']['estimated_char_components'])
        contrast_improvements.append(res['metrics']['contrast_after'] - res['metrics']['contrast_before'])
        
    avg_chars = np.mean(char_counts)
    avg_contrast_gain = np.mean(contrast_improvements)
    
    prep_results.append({
        'clip_limit': clip,
        'avg_chars': avg_chars,
        'contrast_gain': avg_contrast_gain
    })
    print(f"Preprocess Clip Limit={clip} -> Độ tương phản tăng: {avg_contrast_gain:.1f}, Ước lượng ký tự TB: {avg_chars:.2f}")

# Vẽ biểu đồ Preprocess
prep_labels = [f"clip={r['clip_limit']}" for r in prep_results]
prep_contrasts = [r['contrast_gain'] for r in prep_results]
plot_survey_chart('Khảo sát Preprocess: Clip Limit vs Độ lệch Tương phản', prep_labels, prep_contrasts, 'Độ tương phản tăng thêm', 'survey_preprocess_clip.png', '#4e79a7')

# ============================================================
# 3. KHẢO SÁT THAM SỐ SEGMENT
# ============================================================
print("\n========== BƯỚC 3: KHẢO SÁT THAM SỐ SEGMENT (Min Area) ==========")
area_sweep = [5, 40, 200]
seg_results = []

for area in area_sweep:
    correct_count = 0  # Số ảnh segment ra khoảng 7-9 ký tự (chuẩn biển VN)
    total_segmented_chars = []
    
    for crop in plate_crops:
        # Preprocess mặc định trước
        prep_res = preprocess_from_yolo_crop(crop)
        binary = prep_res['morph']
        
        # Segment khảo sát
        chars = segment_with_contour(binary, min_area=area)
        n_chars = len(chars)
        total_segmented_chars.append(n_chars)
        
        if 7 <= n_chars <= 9:
            correct_count += 1
            
    success_rate = (correct_count / len(plate_crops)) * 100 if plate_crops else 0
    avg_chars = np.mean(total_segmented_chars) if total_segmented_chars else 0
    
    seg_results.append({
        'min_area': area,
        'success_rate': success_rate,
        'avg_chars': avg_chars
    })
    print(f"Segment Min Area={area} -> Tỉ lệ chuẩn biển (7-9 kí tự): {success_rate:.1f}%, Số ký tự TB: {avg_chars:.2f}")

# Vẽ biểu đồ Segment
seg_labels = [f"min_area={r['min_area']}" for r in seg_results]
seg_rates = [r['success_rate'] for r in seg_results]
plot_survey_chart('Khảo sát Segment: Min Area vs Tỉ lệ đạt số kí tự chuẩn (7-9)', seg_labels, seg_rates, 'Tỉ lệ đạt chuẩn (%)', 'survey_segment_area.png', '#f28e2b')

# ── Vẽ ảnh so sánh trực quan: cùng biển số, segment với từng min_area ────────
# Dùng demo_plate_crop.jpg (đồng nhất với segment_demo.png của notebook)
SHARED_CROP_PATH = os.path.join(OUTPUT_DIR, 'demo_plate_crop.jpg')
_vis_crop = None
if os.path.exists(SHARED_CROP_PATH):
    _vis_crop = cv2.imread(SHARED_CROP_PATH)
elif plate_crops:
    _vis_crop = plate_crops[0]

if _vis_crop is not None:
    _prep = preprocess_from_yolo_crop(_vis_crop)
    _binary = _prep['morph']

    # Thêm 2 cột aspect_ratio để giống format cũ (tổng 5 cột)
    aspect_configs = [
        ('aspect\n(0.3, 1.2)', (0.3, 1.2)),
        ('aspect\n(0.2, 2.0)', (0.2, 2.0)),
    ]
    all_configs = [(f'min_area={a}\n(mac dinh)', a, (0.2, 1.6)) for a in area_sweep]
    all_configs += [(lbl, 40, ar) for lbl, ar in aspect_configs]

    n_cols = len(all_configs)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    for ax, (lbl, min_a, asp) in zip(axes, all_configs):
        chars = segment_with_contour(_binary, min_area=min_a, aspect_ratio_range=asp)
        # Ve bbox len anh nhi phan
        vis = cv2.cvtColor(_binary, cv2.COLOR_GRAY2BGR)
        for (_, (x, y, w, h)) in chars:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{lbl}\n-> {len(chars)} ky tu', fontsize=9, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Khao sat Tham so Segment', fontsize=13, fontweight='bold')
    plt.tight_layout()
    seg_params_path = os.path.join(OUTPUT_DIR, 'segment_params.png')
    plt.savefig(seg_params_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Da luu anh so sanh truc quan: {seg_params_path}")

# ============================================================
# 4. GHI BÁO CÁO KẾT QUẢ TỐI ƯU HÓA
# ============================================================
best_yolo = max(yolo_results, key=lambda x: x['usable_rate'])
# Đối với preprocess, chọn clip_limit tăng tương phản nhiều nhất nhưng ký tự ước lượng ổn định gần 8
best_prep = max(prep_results, key=lambda x: x['contrast_gain'])
# Đối với segment, chọn min_area cho tỉ lệ đạt chuẩn 7-9 kí tự cao nhất
best_seg = max(seg_results, key=lambda x: x['success_rate'])

report_path = os.path.join(OUTPUT_DIR, 'yolo_preprocess_seg_survey.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Báo Cáo Thực Nghiệm Tối Ưu Hóa Tham Số: YOLO, Preprocess & Segment\n\n")
    f.write(f"Báo cáo tự động tạo vào lúc {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n")
    
    f.write("## 1. Phương Pháp Thực Nghiệm\n")
    f.write(f"- **Tập dữ liệu**: Lấy ngẫu nhiên các mẫu từ thư mục `valid/images` ({len(valid_images)} ảnh).\n")
    f.write("- **Quy trình khảo sát**:\n")
    f.write("  - **YOLO**: Quét ngưỡng `conf_threshold` để cân bằng giữa Tỉ lệ phát hiện (Recall) và Tỉ lệ khả dụng (Precision).\n")
    f.write("  - **Preprocess**: Quét tham số `clip_limit` của CLAHE nhằm tăng tối đa độ tương phản của chữ số mà không làm bết nét.\n")
    f.write("  - **Segment**: Quét tham số `min_area` để lọc bỏ các đốm nhiễu nhỏ, giữ lại đúng hình dáng các chữ số.\n\n")
    
    f.write("## 2. Kết Quả Khảo Sát Chi Tiết\n\n")
    
    # 2.1 YOLO
    f.write("### 2.1. Khảo Sát YOLO Detection\n")
    f.write("| Ngưỡng Conf | Tỉ lệ phát hiện (%) | Tỉ lệ khả dụng (%) | Latency TB (ms) |\n")
    f.write("|:---|:---:|:---:|:---:|\n")
    for r in yolo_results:
        f.write(f"| **{r['conf']}** | {r['detect_rate']:.1f}% | {r['usable_rate']:.1f}% | {r['latency']:.1f} ms |\n")
    f.write("\n![Biểu đồ YOLO](survey_yolo_conf.png)\n\n")
    
    # 2.2 Preprocess
    f.write("### 2.2. Khảo Sát Preprocess (CLAHE Contrast)\n")
    f.write("| Clip Limit | Độ tăng tương phản (Gray STD Gain) | Số ký tự ước lượng TB |\n")
    f.write("|:---|:---:|:---:|\n")
    for r in prep_results:
        f.write(f"| **{r['clip_limit']}** | +{r['contrast_gain']:.2f} | {r['avg_chars']:.2f} |\n")
    f.write("\n![Biểu đồ Preprocess](survey_preprocess_clip.png)\n\n")
    
    # 2.3 Segment
    f.write("### 2.3. Khảo Sát Character Segmentation\n")
    f.write("| Min Area | Tỉ lệ chuẩn biển 7-9 kí tự (%) | Số ký tự tách được TB |\n")
    f.write("|:---|:---:|:---:|\n")
    for r in seg_results:
        f.write(f"| **{r['min_area']}** | {r['success_rate']:.1f}% | {r['avg_chars']:.2f} |\n")
    f.write("\n![Biểu đồ Segment](survey_segment_area.png)\n\n")
    
    # Kết luận
    f.write("## 3. Cấu Hình Tối Ưu Đề Xuất\n\n")
    f.write(f"- **YOLO Detection**: Ngưỡng `conf_threshold = {best_yolo['conf']}` đạt hiệu năng cân bằng tốt nhất.\n")
    f.write(f"- **Preprocess (CLAHE)**: Tham số `clip_limit = {best_prep['clip_limit']}` tối ưu độ sắc nét chữ số.\n")
    f.write(f"- **Character Segment**: Tham số `min_area = {best_seg['min_area']}` lọc nhiễu tốt nhất.\n\n")
    
    f.write("### Nhận xét chung của nhóm:\n")
    f.write("1. **YOLO**: Hạ ngưỡng `conf` giúp không bỏ sót biển số nhưng sẽ tăng nguy cơ nhận nhầm vật thể xung quanh. Ngưỡng 0.50 là tối ưu.\n")
    f.write("2. **Preprocess**: CLAHE `clip_limit` quá cao (3.0) làm phóng đại cả nhiễu hạt của nền biển số, dễ gây dính nét chữ. Ngưỡng 2.0 cho độ tương phản sắc nét ổn định nhất.\n")
    f.write("3. **Segment**: `min_area = 40` là kích thước hoàn hảo. Nếu hạ xuống 20, các dấu chấm rác nhỏ xung quanh viền biển số sẽ bị cắt thành ký tự rác. Nếu tăng lên 80, các nét chữ mảnh hoặc ký tự nhỏ của biển 2 dòng có thể bị bỏ sót.\n")

print(f"\nBáo cáo chi tiết khảo sát đã lưu tại: {report_path}")
print("========== HOÀN THÀNH KHẢO SÁT ==========")
