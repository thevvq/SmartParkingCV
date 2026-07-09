"""
Module phát hiện và cắt biển số bằng YOLO.

Tác giả: [Tên bạn]
Mô tả: Module này cung cấp lớp YOLODetector để phát hiện biển số xe 
       và crop vùng biển số từ ảnh đầu vào. Hỗ trợ khảo sát tham số 
       và đánh giá trên thư mục ảnh.
"""
import cv2
import numpy as np
import time
import os
import pandas as pd
from ultralytics import YOLO


class YOLODetector:
    """
    Lớp phát hiện và cắt biển số bằng YOLO.
    
    Attributes:
        model (YOLO): Model YOLO đã load.
        conf_threshold (float): Ngưỡng confidence.
        iou_threshold (float): Ngưỡng IoU cho NMS.
        imgsz (int): Kích thước ảnh đầu vào.
    """
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, 
                 iou_threshold=0.45, imgsz=640):
        """
        Khởi tạo detector.
        
        Args:
            model_path (str): Đường dẫn đến model YOLO (.pt).
            conf_threshold (float): Ngưỡng confidence (0-1).
            iou_threshold (float): Ngưỡng IoU cho NMS (0-1).
            imgsz (int): Kích thước ảnh đầu vào.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        
    def detect(self, image):
        """
        Phát hiện vật thể trong ảnh.
        
        Args:
            image (numpy.ndarray): Ảnh đầu vào (BGR).
            
        Returns:
            list: Danh sách các detection, mỗi detection là dict:
                - 'bbox': (x1, y1, x2, y2) tọa độ pixel
                - 'confidence': float độ tin cậy
                - 'class': int id lớp (0 = license_plate)
        """
        # Chạy YOLO inference
        results = self.model(
            image, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold, 
            imgsz=self.imgsz,
            verbose=False  # Tắt log để gọn
        )
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Lấy tọa độ (đã chuyển về pixel)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': cls
                    })
        return detections
    
    def crop_plate(self, image, padding_ratio=0.05):
        """
        Crop vùng biển số với padding.
        
        Args:
            image (numpy.ndarray): Ảnh đầu vào (BGR).
            padding_ratio (float): Tỉ lệ mở rộng bbox (0-1).
            
        Returns:
            tuple: (cropped_image, confidence) hoặc (None, 0.0) nếu không detect.
        """
        detections = self.detect(image)
        if not detections:
            return None, 0.0
            
        # Chọn detection có confidence cao nhất
        best = max(detections, key=lambda d: d['confidence'])
        x1, y1, x2, y2 = best['bbox']
        h, w = image.shape[:2]
        
        # Tính padding
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
        
        # Áp dụng padding, giới hạn trong biên ảnh
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        cropped = image[y1:y2, x1:x2]
        return cropped, best['confidence']
    
    def detect_and_draw(self, image):
        """
        Vẽ bounding box lên ảnh và trả về ảnh đã vẽ.
        
        Args:
            image (numpy.ndarray): Ảnh đầu vào.
            
        Returns:
            numpy.ndarray: Ảnh đã vẽ bbox và confidence.
        """
        detections = self.detect(image)
        img_copy = image.copy()
        
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            conf = d['confidence']
            
            # Vẽ bbox
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Vẽ label với confidence
            label = f"{conf:.2f}"
            cv2.putText(img_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return img_copy
    
    def evaluate_on_folder(self, folder_path, padding_ratio=0.05, 
                           save_crops=False, crop_dir=None):
        """
        Đánh giá YOLO trên một thư mục ảnh.
        
        Args:
            folder_path (str): Đường dẫn thư mục chứa ảnh.
            padding_ratio (float): Tỉ lệ padding cho crop.
            save_crops (bool): Có lưu ảnh crop không.
            crop_dir (str): Thư mục lưu ảnh crop (nếu save_crops=True).
            
        Returns:
            dict: Kết quả đánh giá:
                - detection_rate: Tỉ lệ phát hiện (%)
                - usable_rate: Tỉ lệ crop dùng được (%)
                - avg_confidence: Confidence trung bình
                - avg_latency: Thời gian xử lý trung bình (ms)
                - details: DataFrame chi tiết từng ảnh
        """
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"Không tìm thấy ảnh trong {folder_path}")
            return None
            
        results = []
        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Đo thời gian
            start = time.time()
            detections = self.detect(img)
            elapsed = (time.time() - start) * 1000  # ms
            
            has_detect = len(detections) > 0
            cropped = None
            usable = False
            confidence = 0.0
            
            if has_detect:
                cropped, confidence = self.crop_plate(img, padding_ratio)
                if cropped is not None:
                    h, w = cropped.shape[:2]
                    # Tiêu chí usable: kích thước đủ lớn
                    if w > 80 and h > 25:
                        usable = True
                    if save_crops and crop_dir is not None:
                        os.makedirs(crop_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(crop_dir, fname), cropped)
                        
            results.append({
                'file': fname,
                'detected': has_detect,
                'usable': usable,
                'confidence': confidence,
                'latency': elapsed
            })
            
        df = pd.DataFrame(results)
        detection_rate = df['detected'].mean() * 100
        usable_rate = df['usable'].mean() * 100 if df['detected'].sum() > 0 else 0
        avg_conf = df[df['detected']]['confidence'].mean() if df['detected'].sum() > 0 else 0
        avg_latency = df['latency'].mean()
        
        return {
            'detection_rate': detection_rate,
            'usable_rate': usable_rate,
            'avg_confidence': avg_conf,
            'avg_latency': avg_latency,
            'details': df
        }
    
    def grid_search(self, folder_path, param_grid, crop_dir=None):
        """
        Thực hiện grid search để tối ưu tham số.
        
        Args:
            folder_path (str): Thư mục chứa ảnh test.
            param_grid (dict): Các tham số cần khảo sát.
            crop_dir (str): Thư mục lưu crop (optional).
            
        Returns:
            pandas.DataFrame: Kết quả grid search.
        """
        # Lưu tham số hiện tại
        original_conf = self.conf_threshold
        original_iou = self.iou_threshold
        original_imgsz = self.imgsz
        
        results = []
        
        # Lấy các giá trị cần khảo sát
        conf_values = param_grid.get('conf_threshold', [self.conf_threshold])
        iou_values = param_grid.get('iou_threshold', [self.iou_threshold])
        imgsz_values = param_grid.get('imgsz', [self.imgsz])
        pad_values = param_grid.get('padding_ratio', [0.05])
        
        total_combos = len(conf_values) * len(iou_values) * len(imgsz_values) * len(pad_values)
        print(f"Bắt đầu grid search với {total_combos} tổ hợp tham số...")
        
        count = 0
        for conf in conf_values:
            self.conf_threshold = conf
            for iou in iou_values:
                self.iou_threshold = iou
                for imgsz in imgsz_values:
                    self.imgsz = imgsz
                    for pad in pad_values:
                        count += 1
                        print(f"  [{count}/{total_combos}] conf={conf}, iou={iou}, imgsz={imgsz}, pad={pad}")
                        
                        eval_res = self.evaluate_on_folder(
                            folder_path,
                            padding_ratio=pad,
                            save_crops=(crop_dir is not None),
                            crop_dir=crop_dir
                        )
                        
                        if eval_res:
                            results.append({
                                'conf_threshold': conf,
                                'iou_threshold': iou,
                                'imgsz': imgsz,
                                'padding_ratio': pad,
                                'detection_rate': eval_res['detection_rate'],
                                'usable_rate': eval_res['usable_rate'],
                                'avg_confidence': eval_res['avg_confidence'],
                                'avg_latency': eval_res['avg_latency']
                            })
        
        # Khôi phục tham số
        self.conf_threshold = original_conf
        self.iou_threshold = original_iou
        self.imgsz = original_imgsz
        
        df = pd.DataFrame(results)
        return df.sort_values('usable_rate', ascending=False)


# ============ HÀM TIỆN ÍCH ĐỂ SO SÁNH VỚI PHƯƠNG PHÁP CỔ ĐIỂN ============

def detect_plate_canny_contour(image, debug=False):
    """
    Phát hiện biển số bằng Canny + Contour (phương pháp cổ điển).
    
    Args:
        image (numpy.ndarray): Ảnh đầu vào (BGR).
        debug (bool): Hiển thị ảnh trung gian.
        
    Returns:
        tuple: (cropped_image, contour) hoặc (None, None) nếu không tìm thấy.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Tìm contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contour theo diện tích và tỉ lệ khung hình (biển số thường có tỉ lệ ~2:1 đến 4:1)
    best_contour = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 50000:  # Lọc diện tích
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        
        # Biển số thường có tỉ lệ 2:1 đến 4:1
        if 2.0 < aspect_ratio < 5.0:
            # Tính điểm dựa trên diện tích và tỉ lệ
            score = area * (1 + abs(aspect_ratio - 3.0) / 3.0)
            if score > best_score:
                best_score = score
                best_contour = cnt
    
    if best_contour is None:
        return None, None
        
    x, y, w, h = cv2.boundingRect(best_contour)
    cropped = image[y:y+h, x:x+w]
    
    if debug:
        # Vẽ contour lên ảnh
        img_contour = image.copy()
        cv2.drawContours(img_contour, [best_contour], -1, (0, 255, 0), 2)
        cv2.imshow('Contour Detection', img_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return cropped, best_contour


def compare_yolo_vs_classical(test_dir, yolo_detector, num_samples=10):
    """
    So sánh YOLO và phương pháp cổ điển trên cùng tập ảnh.
    
    Args:
        test_dir (str): Thư mục ảnh test.
        yolo_detector (YOLODetector): Detector YOLO đã khởi tạo.
        num_samples (int): Số ảnh test (lấy ngẫu nhiên).
        
    Returns:
        pandas.DataFrame: Bảng so sánh kết quả.
    """
    import random
    
    images = [f for f in os.listdir(test_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) > num_samples:
        images = random.sample(images, num_samples)
    
    results = []
    
    for fname in images:
        img_path = os.path.join(test_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # YOLO
        yolo_crop, yolo_conf = yolo_detector.crop_plate(img)
        yolo_success = yolo_crop is not None
        
        # Canny + Contour
        classical_crop, _ = detect_plate_canny_contour(img)
        classical_success = classical_crop is not None
        
        results.append({
            'file': fname,
            'YOLO_success': yolo_success,
            'Classical_success': classical_success,
            'YOLO_confidence': yolo_conf if yolo_success else 0
        })
    
    df = pd.DataFrame(results)
    
    # Tính tỉ lệ thành công
    yolo_rate = df['YOLO_success'].mean() * 100
    classical_rate = df['Classical_success'].mean() * 100
    
    print(f"YOLO detection rate: {yolo_rate:.1f}%")
    print(f"Classical detection rate: {classical_rate:.1f}%")
    print(f"Cải thiện: {yolo_rate - classical_rate:.1f}%")
    
    return df