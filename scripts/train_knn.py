import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knn_classifier import train_knn_model

if __name__ == "__main__":
    # 1. Đường dẫn thư mục dữ liệu và nơi lưu mô hình
    dataset_path = 'dataset/knn'
    model_save_path = 'models/knn_hog_model.joblib'
    
    print("========== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH KNN ==========")
    print(f"Thư mục dữ liệu: {dataset_path}")
    print(f"Đường dẫn lưu mô hình: {model_save_path}")
    
    # 2. Gọi hàm huấn luyện
    # Lưu ý: n_neighbors mặc định đã được set là 1 trong src/knn_classifier.py sau khi khảo sát
    try:
        model = train_knn_model(
            train_path=dataset_path, 
            model_save_path=model_save_path
        )
        print("========== HUẤN LUYỆN THÀNH CÔNG ==========")
    except Exception as e:
        print(f"========== LỖI HUẤN LUYỆN ==========\n{e}")
