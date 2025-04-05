import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageProcessor:
    """
    Lớp xử lý ảnh tập trung cho toàn bộ dự án
    """
    def __init__(self, img_size=224):
        self.img_size = img_size
        # Chuẩn hóa theo ImageNet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess_input(self, x):
        """
        Tiền xử lý ảnh input theo chuẩn của ImageNet
        
        Args:
            x: Ảnh đầu vào
            
        Returns:
            Ảnh đã được chuẩn hóa
        """
        x = x.astype('float32')
        # Chuẩn hóa theo ImageNet
        x = x / 255.0
        x = (x - self.mean) / self.std
        return x
    
    def load_and_preprocess_image(self, image_path, apply_clahe=False):
        """
        Load và tiền xử lý một ảnh đơn lẻ
        
        Args:
            image_path: Đường dẫn đến file ảnh
            apply_clahe: Có áp dụng CLAHE để cải thiện độ tương phản không
            
        Returns:
            Ảnh đã được tiền xử lý
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh từ {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Áp dụng CLAHE nếu được yêu cầu
            if apply_clahe:
                img = apply_clahe(img)
                
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = self.preprocess_input(img)
            return img
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
            raise
    
    def load_and_preprocess_for_prediction(self, image_path, apply_clahe=False):
        """
        Load và tiền xử lý ảnh cho việc dự đoán
        
        Args:
            image_path: Đường dẫn đến file ảnh
            apply_clahe: Có áp dụng CLAHE để cải thiện độ tương phản không
            
        Returns:
            Ảnh đã được tiền xử lý và mở rộng chiều batch
        """
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            # Sử dụng phương thức load_and_preprocess_image đã cập nhật
            img_array = self.load_and_preprocess_image(image_path, apply_clahe=apply_clahe)
            img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
            
            return img_array
            
        except Exception as e:
            print(f"Error in load_and_preprocess_for_prediction: {str(e)}")
            raise

# Thêm các hàm tiện ích cho việc xử lý ảnh hàng loạt
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) để cải thiện độ tương phản
    
    Args:
        image: Ảnh đầu vào
        clip_limit: Giới hạn clip
        tile_grid_size: Kích thước lưới
        
    Returns:
        Ảnh đã được cải thiện độ tương phản
    """
    if len(image.shape) == 3:
        # Chuyển sang không gian màu LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # Tách các kênh
        l, a, b = cv2.split(lab)
        # Áp dụng CLAHE cho kênh L
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        # Hợp nhất các kênh
        limg = cv2.merge((cl, a, b))
        # Chuyển trở lại không gian màu RGB
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        # Áp dụng trực tiếp cho ảnh grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def add_noise(image, noise_type="gaussian", amount=0.05):
    """
    Thêm nhiễu vào ảnh để tăng cường dữ liệu
    
    Args:
        image: Ảnh đầu vào
        noise_type: Loại nhiễu ("gaussian", "salt_pepper")
        amount: Mức độ nhiễu
        
    Returns:
        Ảnh đã được thêm nhiễu
    """
    image_copy = image.copy()
    
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        sigma = amount * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image_copy + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = amount
        out = np.copy(image_copy)
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                 for i in image.shape]
        out[coords[0], coords[1], :] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                 for i in image.shape]
        out[coords[0], coords[1], :] = 0
        
        return out
    
    return image_copy

# Thêm import os ở đầu file
import os
