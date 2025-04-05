import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

def plot_training_history(history):
    """Vẽ đồ thị quá trình huấn luyện với style đẹp hơn"""
    # Thiết lập style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Tạo figure với kích thước lớn hơn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Lấy dữ liệu từ history
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # Plot accuracy với style đẹp hơn
    ax1.plot(epochs, history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim([0, 1.0])  # Giới hạn trục y từ 0 đến 1
    
    # Plot loss với style đẹp hơn
    ax2.plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    ax2.plot(epochs, history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Thêm đường kẻ ngang để dễ so sánh
    ax1.axhline(y=0.8, color='g', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.3)
    
    # Thêm thông tin về khoảng cách giữa training và validation
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    acc_diff = train_acc - val_acc
    
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    loss_diff = val_loss - train_loss
    
    # Thêm text thông tin
    ax1.text(0.02, 0.02, f'Accuracy Gap: {acc_diff:.4f}', 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.text(0.02, 0.02, f'Loss Gap: {loss_diff:.4f}', 
             transform=ax2.transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Thêm title chung
    fig.suptitle('Training History', fontsize=16, fontweight='bold', y=1.05)
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output_plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Vẽ confusion matrix với style đẹp hơn"""
    # Thiết lập style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Chuẩn hóa confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Tạo figure với kích thước lớn hơn
    plt.figure(figsize=(15, 15))
    
    # Vẽ heatmap với style đẹp hơn
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                square=True,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Thêm thông tin về accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    plt.text(0.02, 0.02, f'Overall Accuracy: {accuracy:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output_plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """In báo cáo phân loại với format đẹp hơn"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(report)
    print("="*80)
    
    # Lưu báo cáo vào file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output_plots')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

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