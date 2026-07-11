# Nhận diện biển số xe bằng YOLO, Xử lý ảnh và HOG + kNN

## Giới thiệu

Đây là đồ án môn **Xử lý ảnh và Thị giác máy tính**.

Đề tài thực hiện bài toán nhận diện biển số xe từ ảnh đầu vào bằng cách kết hợp mô hình YOLO với các kỹ thuật xử lý ảnh truyền thống và phương pháp học máy HOG + kNN.

---

## Pipeline xử lý

```text
Ảnh xe đầu vào
    │
    ▼
YOLO phát hiện biển số
    │
    ▼
Crop vùng biển số
    │
    ▼
Resize + Grayscale
    │
    ▼
Percentile Stretch + Auto Gamma
    │
    ▼
CLAHE (tăng cường tương phản cục bộ)
    │
    ▼
Bilateral Filter + Unsharp Mask (giảm nhiễu, làm nét)
    │
    ▼
Threshold nhị phân (Otsu)
    │
    ▼
Morphology (làm sạch nhiễu)
    │
    ▼
Segment ký tự (contour-based)
    │
    ▼
HOG trích xuất đặc trưng
    │
    ▼
kNN nhận dạng ký tự
    │
    ▼
Biển số hoàn chỉnh
```

---

## Công nghệ sử dụng

- Python
- Jupyter Notebook
- OpenCV
- Ultralytics YOLOv8
- NumPy
- scikit-learn
- scikit-image
- Matplotlib
- PyTorch
- Roboflow (thu thập dữ liệu)

---

## Cấu trúc thư mục

```text
.
├── pipeline_notebook.ipynb   # Notebook chạy pipeline từng bước
├── data.yaml                 # Cấu hình dataset cho YOLO
├── requirements.txt          # Thư viện cần cài đặt
├── src/                      # Mã nguồn các module
│   ├── yolo_detector.py      #   Phát hiện biển số bằng YOLO
│   ├── preprocess.py         #   Tiền xử lý ảnh (resize, CLAHE, threshold, morphology…)
│   ├── segment.py            #   Tách ký tự bằng contour
│   ├── hog_feature.py        #   Trích xuất đặc trưng HOG
│   └── knn_classifier.py     #   Phân loại ký tự bằng kNN
├── scripts/                  # Script hỗ trợ
│   ├── param_survey.py       #   Khảo sát tham số
│   ├── train_knn.py          #   Huấn luyện mô hình kNN
│   └── yolo_preprocess_seg_survey.py
├── models/                   # Mô hình đã huấn luyện
│   └── knn_hog_model.joblib  #   Mô hình kNN + HOG
├── dataset/                  # Dữ liệu huấn luyện
├── valid/                    # Dữ liệu kiểm thử
├── experiments/              # Thí nghiệm khảo sát
└── README.md
```

---

## Cách chạy

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Mở Notebook

```bash
jupyter notebook pipeline_notebook.ipynb
```

### 3. Chạy lần lượt các cell từ trên xuống dưới.

> **Lưu ý:** Thay đổi đường dẫn ảnh đầu vào tại **Bước 1** trong notebook nếu muốn thử ảnh khác.

---

## Nội dung Notebook

`pipeline_notebook.ipynb` trình bày pipeline theo 13 bước:

| Bước | Tên                                     | Mô tả                                           |
| ---- | ---------------------------------------- | ------------------------------------------------ |
| 0    | Import thư viện và cấu hình             | Nạp các module từ `src/`, khai báo đường dẫn     |
| 1    | Đọc ảnh xe đầu vào                      | Đọc và hiển thị ảnh xe                           |
| 2    | YOLO – Phát hiện biển số                | Dùng YOLOv8 detect vùng biển số                  |
| 3    | Crop vùng biển số                        | Cắt vùng biển số từ ảnh gốc                      |
| 4    | Grayscale                                | Resize + chuyển ảnh sang xám                     |
| 5    | CLAHE                                    | Percentile Stretch → Auto Gamma → CLAHE          |
| 6    | Bilateral Filter + Unsharp Mask          | Giảm nhiễu và làm nét cạnh                       |
| 7    | Threshold nhị phân                       | Nhị phân hóa ảnh bằng Otsu                       |
| 8    | Morphology                               | Làm sạch nhiễu bằng phép toán hình thái          |
| 9    | Segment ký tự                            | Tách từng ký tự bằng contour                     |
| 10   | HOG                                      | Trích xuất đặc trưng HOG cho từng ký tự          |
| 11   | kNN                                      | Nhận dạng ký tự bằng mô hình kNN                 |
| 12   | Biển số hoàn chỉnh                       | Ghép kết quả và hiển thị biển số nhận dạng được   |

---

## Thành viên thực hiện

| Thành viên   | Công việc               |
| ------------ | ----------------------- |
| Thành viên 1 | YOLO                    |
| Thành viên 2 | Tiền xử lý ảnh          |
| Thành viên 3 | Segment ký tự           |
| Thành viên 4 | HOG + kNN               |
| Thành viên 5 | Thực nghiệm và đánh giá |
| Thành viên 6 | Báo cáo và slide        |

---

## Ghi chú

Đồ án được thực hiện phục vụ mục đích học tập trong học phần **Xử lý ảnh và Thị giác máy tính**.
