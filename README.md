# Nhận diện biển số xe bằng YOLO, Xử lý ảnh và HOG + kNN

## Giới thiệu

Đây là đồ án môn **Xử lý ảnh và Thị giác máy tính**.

Đề tài thực hiện bài toán nhận diện biển số xe từ ảnh đầu vào bằng cách kết hợp mô hình YOLO với các kỹ thuật xử lý ảnh truyền thống và phương pháp học máy HOG + kNN.

---

## Pipeline xử lý

```text
Ảnh xe
    │
    ▼
YOLO phát hiện biển số
    │
    ▼
Crop biển số
    │
    ▼
Grayscale
    │
    ▼
CLAHE
    │
    ▼
Gaussian Blur
    │
    ▼
Adaptive Threshold
    │
    ▼
Morphology
    │
    ▼
Segment ký tự
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
- Ultralytics YOLO
- NumPy
- scikit-learn

---

## Cấu trúc thư mục

```text
.
├── notebook.ipynb
├── dataset/
├── models/
├── outputs/
├── requirements.txt
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
jupyter notebook
```

### 3. Chạy lần lượt các cell từ trên xuống dưới.

---

## Nội dung Notebook

Notebook được trình bày theo các phần:

1. Phát biểu mục tiêu
2. Giới thiệu bài toán
3. Chuẩn bị dữ liệu
4. Huấn luyện YOLO
5. Phát hiện biển số
6. Tiền xử lý ảnh
7. Segment ký tự
8. Trích xuất đặc trưng bằng HOG
9. Huấn luyện và nhận dạng bằng kNN
10. Đánh giá kết quả
11. Kết luận

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
