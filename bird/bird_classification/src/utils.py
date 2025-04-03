import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

def plot_training_history(history):
    """Vẽ đồ thị quá trình huấn luyện"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """In báo cáo phân loại chi tiết"""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def predict_single_image(model, image_path, class_names, img_size=224, use_clahe=False):
    """Dự đoán loài chim cho một ảnh đơn lẻ
    
    Args:
        model: Model đã được huấn luyện
        image_path: Đường dẫn đến ảnh cần dự đoán
        class_names: Danh sách tên các lớp
        img_size: Kích thước ảnh đầu vào cho model
        use_clahe: Có sử dụng CLAHE để cải thiện độ tương phản không
        
    Returns:
        tuple: (tên lớp dự đoán, độ tin cậy)
    """
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Sử dụng ImageProcessor để xử lý ảnh
        from image_utils import ImageProcessor
        
        print(f"Loading image from: {image_path}")
        image_processor = ImageProcessor(img_size=img_size)
        img_array = image_processor.load_and_preprocess_image(image_path, apply_clahe=use_clahe)
        img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
        
        # Dự đoán
        print("Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        print(f"Prediction successful. Class: {class_names[predicted_class]}, Confidence: {confidence:.2f}")
        return class_names[predicted_class], confidence
        
    except Exception as e:
        print(f"Error in predict_single_image: {str(e)}")
        raise