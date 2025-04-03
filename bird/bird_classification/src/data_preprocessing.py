import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import pandas as pd
import matplotlib.pyplot as plt
from image_utils import ImageProcessor, apply_clahe, add_noise

class DataPreprocessor:
    def __init__(self, img_size=224, rotation_range=30, width_shift_range=0.2,
                 height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                 horizontal_flip=True, vertical_flip=False, brightness_range=None,
                 fill_mode='nearest', validation_split=0.2, use_clahe=False):
        """
        Khởi tạo DataPreprocessor
        
        Args:
            img_size: Kích thước ảnh đầu vào
            rotation_range: Phạm vi xoay ảnh (độ)
            width_shift_range: Phạm vi dịch chuyển ngang
            height_shift_range: Phạm vi dịch chuyển dọc
            shear_range: Phạm vi cắt xén
            zoom_range: Phạm vi zoom
            horizontal_flip: Lật ngang
            vertical_flip: Lật dọc (Mới)
            brightness_range: Phạm vi điều chỉnh độ sáng (Mới)
            fill_mode: Chế độ điền
            validation_split: Tỷ lệ dữ liệu validation
            use_clahe: Sử dụng CLAHE để cải thiện độ tương phản (Mới)
        """
        self.img_size = img_size
        self.use_clahe = use_clahe
        self.image_processor = ImageProcessor(img_size=img_size)
        
        # Tạo data generator với các tham số tăng cường dữ liệu
        self.data_gen = ImageDataGenerator(
            preprocessing_function=self.preprocess_input,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            brightness_range=brightness_range,
            fill_mode=fill_mode,
            validation_split=validation_split
        )
        
        # Data generator không có augmentation cho test (Mới)
        self.test_gen = ImageDataGenerator(
            preprocessing_function=self.preprocess_input
        )
    
    def preprocess_input(self, x):
        """Tiền xử lý ảnh input theo chuẩn của ImageNet (Sử dụng ImageProcessor)"""
        # Sử dụng CLAHE nếu được yêu cầu
        if self.use_clahe and len(x.shape) == 3:
            x = apply_clahe(x.astype(np.uint8))  # Convert to uint8 for CLAHE
            
        # Sử dụng image_processor để chuẩn hóa ảnh
        return self.image_processor.preprocess_input(x)
    
    def create_data_generators(self, data_dir, batch_size=32, compute_weights=True):
        """
        Tạo data generators cho training và validation
        
        Args:
            data_dir: Thư mục chứa dữ liệu
            batch_size: Kích thước batch
            compute_weights: Tính toán trọng số lớp để xử lý dữ liệu không cân bằng (Mới)
            
        Returns:
            train_generator, val_generator, class_weights (nếu compute_weights=True)
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"Thư mục dữ liệu không tồn tại: {data_dir}")
        
        print(f"Tạo generators từ thư mục: {data_dir}")
        print(f"Kích thước batch: {batch_size}")
        
        # Tạo train generator
        train_generator = self.data_gen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Tạo validation generator
        val_generator = self.data_gen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        print(f"Số lượng classes: {len(train_generator.class_indices)}")
        print(f"Số lượng ảnh training: {train_generator.samples}")
        print(f"Số lượng ảnh validation: {val_generator.samples}")
        
        # Tính toán class weights nếu được yêu cầu (Mới)
        class_weights = None
        if compute_weights:
            class_weights = self._compute_class_weights(data_dir, train_generator.class_indices)
            print("Class weights:", class_weights)
            
            # Vẽ phân phối lớp (Mới)
            self._plot_class_distribution(data_dir, train_generator.class_indices)
            
        if compute_weights:
            return train_generator, val_generator, class_weights
        else:
            return train_generator, val_generator

    def create_test_generator(self, test_dir, batch_size=32):
        """
        Tạo data generator cho tập test
        
        Args:
            test_dir: Thư mục chứa dữ liệu test
            batch_size: Kích thước batch
            
        Returns:
            test_generator
        """
        if not os.path.exists(test_dir):
            raise ValueError(f"Thư mục test không tồn tại: {test_dir}")
        
        print(f"Tạo test generator từ thư mục: {test_dir}")
        
        test_generator = self.test_gen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Số lượng ảnh test: {test_generator.samples}")
        return test_generator

    def _compute_class_weights(self, data_dir, class_indices):
        """
        Tính toán trọng số cho các lớp dựa trên số lượng mẫu
        
        Args:
            data_dir: Thư mục chứa dữ liệu
            class_indices: Dict ánh xạ tên lớp với chỉ số
            
        Returns:
            Dict trọng số cho mỗi lớp
        """
        class_labels = []
        for class_name, class_idx in class_indices.items():
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                num_samples = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
                class_labels.extend([class_idx] * num_samples)

        if not class_labels:
             print("Warning: No images found in the data directory to compute class weights.")
             return None

        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(class_labels),
            y=class_labels
        )
        
        # Chuyển đổi sang dict
        return {idx: weight for idx, weight in enumerate(class_weights_array)}

    def _plot_class_distribution(self, data_dir, class_indices):
        """
        Vẽ biểu đồ phân phối lớp
        
        Args:
            data_dir: Thư mục chứa dữ liệu
            class_indices: Dict ánh xạ tên lớp với chỉ số
        """
        # Đếm số lượng mẫu cho mỗi lớp
        class_counts = {}
        for class_name in class_indices.keys():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                            if os.path.isfile(os.path.join(class_dir, f))])
                class_counts[class_name] = count
        
        if not class_counts:
            print("Warning: No classes found to plot distribution.")
            return
            
        # Tạo DataFrame
        df = pd.DataFrame({
            'Class': list(class_counts.keys()),
            'Count': list(class_counts.values())
        })
        
        # Sắp xếp theo số lượng giảm dần
        df = df.sort_values('Count', ascending=False)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(15, 8))
        plt.bar(df['Class'], df['Count'])
        plt.xticks(rotation=90)
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution')
        plt.tight_layout()
        
        # Tạo thư mục output nếu chưa có
        output_dir = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'output_plots')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path = os.path.join(output_dir, 'class_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Biểu đồ phân phối lớp đã được lưu tại: {plot_path}")

    def load_and_preprocess_dataset(self, data_dir):
        """Load và tiền xử lý toàn bộ dataset (Sử dụng ImageProcessor)"""
        images = []
        labels = []
        # Ensure class names are sorted consistently
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        class_indices = {name: i for i, name in enumerate(class_names)}
        
        print(f"Đang load dataset từ: {data_dir}")
        print(f"Tìm thấy {len(class_names)} lớp: {class_names}")
        
        for class_name in class_names:
            class_idx = class_indices[class_name]
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            print(f"Đang xử lý lớp {class_name} ({class_idx + 1}/{len(class_names)})")
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                # Check if it's a file before processing
                if os.path.isfile(img_path):
                    try:
                        # Use ImageProcessor to load and preprocess
                        img = self.image_processor.load_and_preprocess_image(img_path)
                        images.append(img)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Bỏ qua ảnh {img_path}: {str(e)}")
        
        return np.array(images), np.array(labels), class_names