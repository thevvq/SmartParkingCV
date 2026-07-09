import os
import cv2
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.knn_classifier import load_images_from_folder
# Cấu hình encoding cho console Windows để hiển thị tiếng Việt
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
# Cấu hình đường dẫn và tham số
DATASET_PATH = 'dataset/knn'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 1. Tải và tiền xử lý dữ liệu
print("========== BƯỚC 1: TẢI DỮ LIỆU ==========")
print(f"Đang tải dữ liệu từ thư mục '{DATASET_PATH}'...")
start_time = time.time()
X, y = load_images_from_folder(DATASET_PATH)
print(f"Tải thành công: {len(X)} ảnh.")
print(f"Thời gian tải dữ liệu: {time.time() - start_time:.2f} giây.")
# Phân chia tập huấn luyện và tập kiểm thử (80% Train, 20% Test, phân lớp Stratified)
print("Đang chia dữ liệu thành tập huấn luyện (80%) và kiểm thử (20%)...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Kích thước tập huấn luyện: {X_train_raw.shape}")
print(f"Kích thước tập kiểm thử: {X_test_raw.shape}")
print(f"Số lượng lớp phân loại: {len(np.unique(y))}")
# 2. Định nghĩa các tham số khảo sát
# Cấu hình cơ sở (baseline) mặc định
baseline_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'knn_k': 5
}
# Các tập giá trị khảo sát
orientations_sweep = [6, 9, 12]
cell_size_sweep = [(4, 4), (8, 8), (16, 16)]
block_size_sweep = [(1, 1), (2, 2), (3, 3)]
knn_k_sweep = [1, 3, 5, 7, 9]
# Hàm trích xuất đặc trưng HOG cho toàn tập dữ liệu
def extract_hog_features_dataset(images, orientations, pixels_per_cell, cells_per_block):
    features = []
    for img in images:
        feat = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False
        )
        features.append(feat)
    return np.array(features)
# Hàm đánh giá một cấu hình tham số
def evaluate_config(orientations, pixels_per_cell, cells_per_block, knn_k):
    start = time.time()
    
    # Trích xuất đặc trưng
    X_train_hog = extract_hog_features_dataset(X_train_raw, orientations, pixels_per_cell, cells_per_block)
    X_test_hog = extract_hog_features_dataset(X_test_raw, orientations, pixels_per_cell, cells_per_block)
    
    # Huấn luyện kNN
    model = KNeighborsClassifier(n_neighbors=knn_k)
    model.fit(X_train_hog, y_train)
    
    # Dự đoán và tính toán độ chính xác
    y_pred = model.predict(X_test_hog)
    accuracy = accuracy_score(y_test, y_pred) * 100
    macro_f1 = f1_score(y_test, y_pred, average='macro') * 100
    
    elapsed = time.time() - start
    return accuracy, macro_f1, elapsed
# Hàm vẽ biểu đồ đôi thể hiện Accuracy và Macro-F1
def plot_survey_results(param_name, labels, accuracies, f1_scores, filename):
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Tông màu hiện đại
    color_acc = '#2b7bba'  # Steel Blue
    color_f1 = '#f28e2b'   # Muted Orange
    
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color=color_acc, alpha=0.9, edgecolor='black', linewidth=0.7)
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Macro-F1 (%)', color=color_f1, alpha=0.9, edgecolor='black', linewidth=0.7)
    
    ax.set_ylabel('Giá trị (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Biểu đồ so sánh: {param_name}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, shadow=False, borderpad=1, fontsize=10)
    
    # Hiển thị số liệu dạng chữ trên đầu mỗi cột
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
                        
    add_labels(rects1)
    add_labels(rects2)
    
    fig.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
# 3. Tiến hành khảo sát
results = {}
# Chạy cấu hình baseline trước để tái sử dụng
print("\n========== BƯỚC 2: CHẠY CẤU HÌNH THAM CHIẾU (BASELINE) ==========")
print(f"Cấu hình baseline: {baseline_params}")
base_acc, base_f1, base_time = evaluate_config(
    baseline_params['orientations'],
    baseline_params['pixels_per_cell'],
    baseline_params['cells_per_block'],
    baseline_params['knn_k']
)
print(f"Accuracy: {base_acc:.2f}%, Macro-F1: {base_f1:.2f}% (Thời gian chạy: {base_time:.2f}s)")
# --- KHẢO SÁT 1: HOG Orientations ---
print("\n========== KHẢO SÁT 1: HOG Orientations ==========")
orientations_results = []
for val in orientations_sweep:
    if val == baseline_params['orientations']:
        orientations_results.append((val, base_acc, base_f1))
        print(f"Orientations = {val} (Baseline): Accuracy = {base_acc:.2f}%, F1 = {base_f1:.2f}%")
    else:
        print(f"Đang chạy Orientations = {val}...")
        acc, f1, t = evaluate_config(val, baseline_params['pixels_per_cell'], baseline_params['cells_per_block'], baseline_params['knn_k'])
        orientations_results.append((val, acc, f1))
        print(f"Orientations = {val}: Accuracy = {acc:.2f}%, F1 = {f1:.2f}% (Thời gian: {t:.2f}s)")
results['orientations'] = orientations_results
# --- KHẢO SÁT 2: HOG Cell Size ---
print("\n========== KHẢO SÁT 2: HOG Cell Size ==========")
cell_size_results = []
for val in cell_size_sweep:
    if val == baseline_params['pixels_per_cell']:
        cell_size_results.append((val, base_acc, base_f1))
        print(f"Cell Size = {val} (Baseline): Accuracy = {base_acc:.2f}%, F1 = {base_f1:.2f}%")
    else:
        print(f"Đang chạy Cell Size = {val}...")
        acc, f1, t = evaluate_config(baseline_params['orientations'], val, baseline_params['cells_per_block'], baseline_params['knn_k'])
        cell_size_results.append((val, acc, f1))
        print(f"Cell Size = {val}: Accuracy = {acc:.2f}%, F1 = {f1:.2f}% (Thời gian: {t:.2f}s)")
results['cell_size'] = cell_size_results
# --- KHẢO SÁT 3: HOG Block Size ---
print("\n========== KHẢO SÁT 3: HOG Block Size ==========")
block_size_results = []
for val in block_size_sweep:
    if val == baseline_params['cells_per_block']:
        block_size_results.append((val, base_acc, base_f1))
        print(f"Block Size = {val} (Baseline): Accuracy = {base_acc:.2f}%, F1 = {base_f1:.2f}%")
    else:
        print(f"Đang chạy Block Size = {val}...")
        acc, f1, t = evaluate_config(baseline_params['orientations'], baseline_params['pixels_per_cell'], val, baseline_params['knn_k'])
        block_size_results.append((val, acc, f1))
        print(f"Block Size = {val}: Accuracy = {acc:.2f}%, F1 = {f1:.2f}% (Thời gian: {t:.2f}s)")
results['block_size'] = block_size_results
# --- KHẢO SÁT 4: kNN (K) ---
print("\n========== KHẢO SÁT 4: kNN (K) ==========")
knn_k_results = []
for val in knn_k_sweep:
    if val == baseline_params['knn_k']:
        knn_k_results.append((val, base_acc, base_f1))
        print(f"kNN K = {val} (Baseline): Accuracy = {base_acc:.2f}%, F1 = {base_f1:.2f}%")
    else:
        print(f"Đang chạy kNN K = {val}...")
        acc, f1, t = evaluate_config(baseline_params['orientations'], baseline_params['pixels_per_cell'], baseline_params['cells_per_block'], val)
        knn_k_results.append((val, acc, f1))
        print(f"kNN K = {val}: Accuracy = {acc:.2f}%, F1 = {f1:.2f}% (Thời gian: {t:.2f}s)")
results['knn_k'] = knn_k_results
# 4. Vẽ các biểu đồ và lưu trữ
print("\n========== BƯỚC 3: VẼ BIỂU ĐỒ VÀ GHI KẾT QUẢ ==========")
# Khảo sát 1: Orientations
orient_labels = [str(r[0]) for r in orientations_results]
orient_accs = [r[1] for r in orientations_results]
orient_f1s = [r[2] for r in orientations_results]
plot_survey_results('HOG Orientations', orient_labels, orient_accs, orient_f1s, 'survey_orientations.png')
# Khảo sát 2: Cell Size
cell_labels = [f"{r[0][0]}x{r[0][1]}" for r in cell_size_results]
cell_accs = [r[1] for r in cell_size_results]
cell_f1s = [r[2] for r in cell_size_results]
plot_survey_results('HOG Cell Size', cell_labels, cell_accs, cell_f1s, 'survey_cell_size.png')
# Khảo sát 3: Block Size
block_labels = [f"{r[0][0]}x{r[0][1]}" for r in block_size_results]
block_accs = [r[1] for r in block_size_results]
block_f1s = [r[2] for r in block_size_results]
plot_survey_results('HOG Block Size', block_labels, block_accs, block_f1s, 'survey_block_size.png')
# Khảo sát 4: kNN K
knn_labels = [f"k = {r[0]}" for r in knn_k_results]
knn_accs = [r[1] for r in knn_k_results]
knn_f1s = [r[2] for r in knn_k_results]
plot_survey_results('kNN (K)', knn_labels, knn_accs, knn_f1s, 'survey_knn_k.png')
# Tìm cấu hình tốt nhất tổng thể
all_runs = []
# orientations
for r in orientations_results:
    all_runs.append((f"Orientations={r[0]}, CellSize=(8,8), BlockSize=(2,2), K=5", r[1], r[2]))
# cell size
for r in cell_size_results:
    if r[0] != baseline_params['pixels_per_cell']:
        all_runs.append((f"Orientations=9, CellSize={r[0]}, BlockSize=(2,2), K=5", r[1], r[2]))
# block size
for r in block_size_results:
    if r[0] != baseline_params['cells_per_block']:
        all_runs.append((f"Orientations=9, CellSize=(8,8), BlockSize={r[0]}, K=5", r[1], r[2]))
# knn k
for r in knn_k_results:
    if r[0] != baseline_params['knn_k']:
        all_runs.append((f"Orientations=9, CellSize=(8,8), BlockSize=(2,2), K={r[0]}", r[1], r[2]))
best_run_acc = max(all_runs, key=lambda x: x[1])
best_run_f1 = max(all_runs, key=lambda x: x[2])
# 5. Ghi kết quả ra file Markdown báo cáo
report_path = os.path.join(OUTPUT_DIR, 'survey_results.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Báo Cáo Thực Nghiệm Khảo Sát Tham Số HOG & kNN\n\n")
    f.write(f"Báo cáo tự động tạo vào lúc {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n")
    f.write("## 1. Phương Pháp Thực Nghiệm\n")
    f.write("- **Tập dữ liệu**: Tải từ `dataset/knn` gồm 31 lớp ký tự (chữ số, chữ cái và lớp nhiễu).\n")
    f.write("- **Chia tập dữ liệu**: Phân chia ngẫu nhiên 80% cho tập huấn luyện và 20% cho tập kiểm thử, áp dụng phân lớp Stratified để giữ nguyên tỷ lệ mẫu của các lớp.\n")
    f.write("- **Kích thước ảnh**: Tự động đưa về độ phân giải chuẩn hóa `48x48` trước khi trích xuất đặc trưng.\n")
    f.write("- **Cấu hình baseline (mặc định)**:\n")
    f.write("  - HOG Orientations: `9`\n")
    f.write("  - HOG Cell Size: `(8, 8)`\n")
    f.write("  - HOG Block Size: `(2, 2)`\n")
    f.write("  - kNN K: `5`\n\n")
    
    # Khảo sát 1
    f.write("## 2. Kết Quả Khảo Sát Từng Nhóm Tham Số\n\n")
    f.write("### 2.1. HOG Orientations\n")
    f.write("Khảo sát số lượng hướng biên độ gradient được chia trong mỗi cell (giữ nguyên Cell Size = (8,8), Block Size = (2,2), kNN K = 5):\n\n")
    f.write("| HOG Orientations | Accuracy (%) | Macro-F1 (%) |\n")
    f.write("|:---|:---:|:---:|\n")
    for r in orientations_results:
        f.write(f"| **{r[0]}** | {r[1]:.2f}% | {r[2]:.2f}% |\n")
    f.write("\n![Biểu đồ HOG Orientations](survey_orientations.png)\n\n")
    
    # Khảo sát 2
    f.write("### 2.2. HOG Cell Size\n")
    f.write("Khảo sát kích thước vùng cell tính toán histogram (giữ nguyên Orientations = 9, Block Size = (2,2), kNN K = 5):\n\n")
    f.write("| HOG Cell Size | Accuracy (%) | Macro-F1 (%) |\n")
    f.write("|:---|:---:|:---:|\n")
    for r in cell_size_results:
        f.write(f"| **{r[0][0]}x{r[0][1]}** | {r[1]:.2f}% | {r[2]:.2f}% |\n")
    f.write("\n![Biểu đồ HOG Cell Size](survey_cell_size.png)\n\n")
    
    # Khảo sát 3
    f.write("### 2.3. HOG Block Size\n")
    f.write("Khảo sát số lượng cell gộp chung vào 1 block để chuẩn hóa độ sáng (giữ nguyên Orientations = 9, Cell Size = (8,8), kNN K = 5):\n\n")
    f.write("| HOG Block Size | Accuracy (%) | Macro-F1 (%) |\n")
    f.write("|:---|:---:|:---:|\n")
    for r in block_size_results:
        f.write(f"| **{r[0][0]}x{r[0][1]}** | {r[1]:.2f}% | {r[2]:.2f}% |\n")
    f.write("\n![Biểu đồ HOG Block Size](survey_block_size.png)\n\n")
    
    # Khảo sát 4
    f.write("### 2.4. Bộ Phân Loại kNN (K)\n")
    f.write("Khảo sát số lượng láng giềng gần nhất k trong bộ phân loại kNN (giữ nguyên Orientations = 9, Cell Size = (8,8), Block Size = (2,2)):\n\n")
    f.write("| kNN (K) | Accuracy (%) | Macro-F1 (%) |\n")
    f.write("|:---|:---:|:---:|\n")
    for r in knn_k_results:
        f.write(f"| **{r[0]}** | {r[1]:.2f}% | {r[2]:.2f}% |\n")
    f.write("\n![Biểu đồ kNN K](survey_knn_k.png)\n\n")
    
    # Kết luận
    f.write("## 3. Kết Luận Và Đề Xuất\n\n")
    f.write(f"- **Cấu hình cho Accuracy cao nhất**: `{best_run_acc[0]}` với **{best_run_acc[1]:.2f}%**\n")
    f.write(f"- **Cấu hình cho Macro-F1 cao nhất**: `{best_run_f1[0]}` với **{best_run_f1[1]:.2f}%**\n\n")
    f.write("### Nhận xét & Đánh giá chi tiết:\n")
    f.write("1. **HOG Orientations**: Số hướng orientations quyết định độ chi tiết của đặc trưng góc nghiêng cạnh. Số lượng orientations lớn hơn có thể tăng biểu diễn thông tin hình dáng chữ nhưng tăng kích thước đặc trưng và độ phức tạp tính toán.\n")
    f.write("2. **HOG Cell Size**: Kích thước cell nhỏ (ví dụ 4x4) giúp thu được nhiều thông tin chi tiết hơn, đạt độ chính xác cao hơn nhưng kích thước đặc trưng tăng lên rất nhiều (làm tăng thời gian dự đoán). Cell lớn (16x16) làm giảm đáng kể thông tin và gây sụt giảm độ chính xác rõ rệt.\n")
    f.write("3. **HOG Block Size**: Kích thước block gộp cell quyết định phạm vi chuẩn hóa tương phản ánh sáng. Block quá bé (1x1) không có tác dụng chuẩn hóa không gian lân cận tốt, block quá to có thể làm mờ biên độ đặc trưng.\n")
    f.write("4. **kNN K**: Giá trị K nhỏ (như K=1) thường nhạy cảm với nhiễu nhưng trong bài toán nhận diện chữ đã chuẩn hóa tốt đôi khi cho kết quả phân lớp chi tiết cao. K lớn hơn giúp đường ranh giới phân loại mượt hơn nhưng có thể làm mờ các lớp có ít mẫu.\n")
print(f"\nBáo cáo chi tiết đã được xuất ra thành công tại: {report_path}")
print("========== HOÀN THÀNH KHẢO SÁT ==========")