import os
import cv2
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from src.hog_feature import compute_hog_features

def load_images_from_folder(folder):
    """
    Tải danh sách ảnh từ thư mục có cấu trúc dạng: folder/label/*.jpg
    Tự động resize ảnh về kích thước 48x48.
    
    Args:
        folder (str): Đường dẫn đến thư mục gốc.
        
    Returns:
        tuple: (mảng_numpy_chứa_ảnh, mảng_numpy_chứa_nhãn)
    """
    images = []
    labels = []
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {folder}")
        
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            # Hỗ trợ cả tên thư mục dạng ký tự (ví dụ: 'A', '0') hoặc mã ASCII (ví dụ: '65', '48')
            if subdir == '999' or subdir.lower() == 'noise':
                label = 999
            elif len(subdir) == 1:
                label = ord(subdir)
            else:
                try:
                    label = int(subdir)
                except ValueError:
                    # Bỏ qua các thư mục không hợp lệ
                    continue

            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label)
                    
    return np.array(images), np.array(labels)

def train_knn_model(train_path, n_neighbors=1, model_save_path=None):
    """
    Huấn luyện bộ phân loại kNN dựa trên các đặc trưng HOG.
    
    Args:
        train_path (str): Đường dẫn đến thư mục chứa dữ liệu huấn luyện.
        n_neighbors (int): Số lượng láng giềng gần nhất k.
        model_save_path (str, optional): Đường dẫn lưu mô hình sau huấn luyện.
        
    Returns:
        KNeighborsClassifier: Mô hình kNN đã huấn luyện.
    """
    print(f"Đang tải dữ liệu huấn luyện từ {train_path}...")
    X_train, y_train = load_images_from_folder(train_path)
    
    if len(X_train) == 0:
        raise ValueError(f"Không tìm thấy ảnh huấn luyện nào trong {train_path}")
        
    print(f"Đang trích xuất đặc trưng HOG cho {len(X_train)} ảnh huấn luyện...")
    X_train_hog = np.array([compute_hog_features(img) for img in X_train])
    
    print(f"Đang huấn luyện mô hình kNN (k={n_neighbors})...")
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train_hog, y_train)
    
    if model_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        joblib.dump(knn_model, model_save_path)
        print(f"Mô hình đã được lưu thành công tại {model_save_path}")
        
    return knn_model

def load_knn_model(model_path):
    """
    Tải mô hình kNN đã được huấn luyện trước đó.
    
    Args:
        model_path (str): Đường dẫn đến file mô hình (.joblib).
        
    Returns:
        KNeighborsClassifier: Mô hình kNN tải lên.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}")
    return joblib.load(model_path)

def decode_label(label):
    """
    Giải mã nhãn số/ASCII sang ký tự hiển thị.
    
    Args:
        label (int): Giá trị mã ASCII hoặc số 999.
        
    Returns:
        str: Ký tự tương ứng hoặc chuỗi rỗng nếu là lớp null.
    """
    if label == 999 or label == 0:
        return ""
    try:
        return chr(label)
    except ValueError:
        return str(label)

def predict_characters(model, char_images):
    """
    Dự đoán nhãn ký tự cho một danh sách ảnh đầu vào.
    
    Args:
        model (KNeighborsClassifier): Mô hình kNN đã huấn luyện.
        char_images (list of numpy.ndarray): Danh sách ảnh ký tự cần nhận diện.
        
    Returns:
        list of str: Danh sách các ký tự nhận dạng được.
    """
    if not char_images:
        return []
        
    hog_features = []
    for img in char_images:
        if img is not None:
            hog_features.append(compute_hog_features(img))
            
    if len(hog_features) == 0:
        return []
        
    hog_features = np.array(hog_features)
    predictions = model.predict(hog_features)
    
    return [decode_label(pred) for pred in predictions]
