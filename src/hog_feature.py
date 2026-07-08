import cv2
import numpy as np
from skimage.feature import hog

# Cấu hình HOG dựa trên kết quả thực nghiệm trong notebook
IMG_HEIGHT = 48
IMG_WIDTH = 48

HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'visualize': False
}

def compute_hog_features(image):
    """
    Tính toán đặc trưng HOG cho một ảnh xám (grayscale) đơn lẻ.
    Nếu ảnh không có kích thước 48x48, nó sẽ được tự động resize.
    
    Args:
        image (numpy.ndarray): Ảnh xám đầu vào.
        
    Returns:
        numpy.ndarray: Vector đặc trưng HOG 1 chiều (kích thước 900).
    """
    if image is None:
        raise ValueError("Ảnh đầu vào không được là None")
        
    # Đảm bảo ảnh là ảnh xám
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Resize về 48x48 (kích thước tối ưu tìm được qua thực nghiệm)
    if image.shape[0] != IMG_HEIGHT or image.shape[1] != IMG_WIDTH:
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        
    # Tính toán đặc trưng HOG
    features = hog(image, **HOG_PARAMS)
    return features

def extract_hog_from_characters(char_images):
    """
    Trích xuất đặc trưng HOG từ danh sách các ảnh ký tự.
    
    Args:
        char_images (list of numpy.ndarray): Danh sách ảnh ký tự.
        
    Returns:
        numpy.ndarray: Mảng 2 chiều chứa đặc trưng HOG, kích thước (N, 900).
    """
    hog_features = [compute_hog_features(img) for img in char_images if img is not None]
    return np.array(hog_features)
