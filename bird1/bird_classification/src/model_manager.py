import os
import tensorflow as tf
import json
from datetime import datetime

class ModelManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, 'models')
        self.model_info_file = os.path.join(self.models_dir, 'model_info.json')
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model(self, model, model_name, training_history=None, class_names=None, img_size=224, use_clahe=False):
        """Lưu model và thông tin liên quan"""
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.h5"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Lưu model
        model.save(model_path)
        
        # Lưu thông tin model
        model_info = {
            'model_name': model_name,
            'timestamp': timestamp,
            'model_path': model_path,
            'class_names': class_names,
            'training_history': training_history,
            'img_size': img_size,
            'use_clahe': use_clahe
        }
        
        # Đọc thông tin cũ nếu có
        existing_info = {}
        if os.path.exists(self.model_info_file):
            with open(self.model_info_file, 'r') as f:
                existing_info = json.load(f)
        
        # Thêm thông tin mới
        existing_info[model_filename] = model_info
        
        # Lưu thông tin mới
        with open(self.model_info_file, 'w') as f:
            json.dump(existing_info, f, indent=4)
        
        return model_path

    def load_model(self, model_name=None):
        """Load model và thông tin liên quan"""
        try:
            if not os.path.exists(self.model_info_file):
                raise FileNotFoundError("No model information file found")
            
            print(f"Reading model info from: {self.model_info_file}")
            # Đọc thông tin model
            with open(self.model_info_file, 'r') as f:
                model_info = json.load(f)
            
            # Nếu không chỉ định model_name, lấy model mới nhất
            if model_name is None:
                model_files = list(model_info.keys())
                if not model_files:
                    raise ValueError("No models found")
                model_name = model_files[-1]  # Lấy model mới nhất
                print(f"Using latest model: {model_name}")
            
            if model_name not in model_info:
                raise ValueError(f"Model {model_name} not found")
            
            # Load model
            model_path = model_info[model_name]['model_path']
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            print(f"Loading model from: {model_path}")
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
            
            if 'class_names' not in model_info[model_name]:
                raise ValueError(f"No class names found for model {model_name}")
            
            return {
                'model': model,
                'info': model_info[model_name]
            }
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def get_available_models(self):
        """Lấy danh sách các model có sẵn"""
        if not os.path.exists(self.model_info_file):
            return []
        
        with open(self.model_info_file, 'r') as f:
            model_info = json.load(f)
        
        return [
            {
                'name': filename,
                'timestamp': info['timestamp'],
                'class_names': info['class_names']
            }
            for filename, info in model_info.items()
        ]