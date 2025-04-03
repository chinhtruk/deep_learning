from flask import Flask, request, render_template, jsonify
import os
from model import BirdClassifier
from utils import predict_single_image
from model_manager import ModelManager
from image_utils import ImageProcessor
import tensorflow as tf
import numpy as np
import traceback

app = Flask(__name__)

# Cấu hình upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'images to predict')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tạo thư mục upload nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Khởi tạo ModelManager
base_dir = os.path.dirname(os.path.dirname(__file__))
model_manager = ModelManager(base_dir)

@app.route('/')
def home():
    try:
        # Lấy danh sách các model có sẵn
        available_models = model_manager.get_available_models()
        if not available_models:
            return render_template('index.html', 
                                 models=[],
                                 error="No trained models found. Please train a model first.")
        return render_template('index.html', models=available_models)
    except Exception as e:
        return render_template('index.html', 
                             models=[],
                             error=f"Error loading models: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Lưu file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        try:
            print(f"Processing file: {filename}")
            
            # Kiểm tra model
            available_models = model_manager.get_available_models()
            if not available_models:
                raise ValueError("No trained models found. Please train a model first.")
            
            print("Loading model...")
            model_data = model_manager.load_model()
            if not model_data or 'model' not in model_data or 'info' not in model_data:
                raise ValueError("Invalid model data structure")
                
            model = model_data['model']
            if model is None:
                raise ValueError("Failed to load model")
                
            class_names = model_data['info']['class_names']
            if not class_names:
                raise ValueError("No class names found in model info")
                
            print(f"Model loaded successfully with {len(class_names)} classes")
            
            # Dự đoán
            print("Starting prediction...")
            try:
                img_size = 224  # Mặc định
                use_clahe = False  # Mặc định

                if 'img_size' in model_data['info']:
                    img_size = model_data['info']['img_size']
                    print(f"Using image size from model: {img_size}")
                
                if 'use_clahe' in model_data['info']:
                    use_clahe = model_data['info']['use_clahe']
                    print(f"Using CLAHE setting from model: {use_clahe}")
                
                predicted_class, confidence = predict_single_image(
                    model=model, 
                    image_path=filename, 
                    class_names=class_names,
                    img_size=img_size,
                    use_clahe=use_clahe
                )
            except Exception as e:
                print(f"Error during image prediction: {str(e)}")
                raise
            
            # Trả kết quả
            response_data = {
                'class': predicted_class,
                'confidence': float(confidence),
                'image_path': filename
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f"Error during prediction: {str(e)}"})
        finally:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    print("File cleaned up successfully")
            except Exception as e:
                print(f"Warning: Could not delete file {filename}: {str(e)}")
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)