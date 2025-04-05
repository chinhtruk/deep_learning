import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data_preprocessing import DataPreprocessor
from model import BirdClassifier
from model_manager import ModelManager
from utils import plot_training_history, plot_confusion_matrix, print_classification_report

def parse_args():
    """Phân tích tham số dòng lệnh"""
    # Sử dụng đường dẫn tuyệt đối thay vì đường dẫn tương đối
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(base_dir, 'data', 'train')
    
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình phân loại chim')
    
    # Tham số dữ liệu
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help='Đường dẫn đến thư mục dữ liệu')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Kích thước ảnh đầu vào (mặc định: 224)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Kích thước batch (mặc định: 32)')
    
    # Tham số tăng cường dữ liệu
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Sử dụng tăng cường dữ liệu')
    parser.add_argument('--use_clahe', action='store_true',
                        help='Sử dụng CLAHE để cải thiện độ tương phản')
    parser.add_argument('--vertical_flip', action='store_true',
                        help='Sử dụng lật dọc trong tăng cường dữ liệu')
    parser.add_argument('--brightness_range', type=float, nargs=2, default=None,
                        help='Phạm vi điều chỉnh độ sáng (ví dụ: 0.8 1.2)')
    
    # Tham số huấn luyện
    parser.add_argument('--model_type', type=str, default='mobilenetv2',
                        choices=['custom_cnn', 'resnet50', 'mobilenetv2'],
                        help='Loại mô hình (mặc định: mobilenetv2)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Số epochs huấn luyện (mặc định: 30)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Tốc độ học (mặc định: 0.001)')
    parser.add_argument('--fine_tune_layers', type=int, default=30,
                        help='Số lớp cuối để fine-tune (chỉ áp dụng cho ResNet50)')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Sử dụng trọng số lớp để xử lý mất cân bằng dữ liệu')
    
    # Tham số lưu trữ
    parser.add_argument('--model_name', type=str, default='bird_classifier',
                        help='Tên mô hình để lưu (mặc định: bird_classifier)')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Cấu hình GPU nếu có
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Đã tìm thấy {len(gpus)} GPU và đã cấu hình thành công.")
        except RuntimeError as e:
            print(f"Lỗi khi cấu hình GPU: {e}")
    else:
        print("Không tìm thấy GPU. Sử dụng CPU.")

    # Tiền xử lý dữ liệu
    print("Khởi tạo bộ tiền xử lý dữ liệu...")
    data_preprocessor = DataPreprocessor(
        img_size=args.img_size,
        rotation_range=30 if args.use_augmentation else 0,
        width_shift_range=0.2 if args.use_augmentation else 0,
        height_shift_range=0.2 if args.use_augmentation else 0,
        shear_range=0.2 if args.use_augmentation else 0,
        zoom_range=0.2 if args.use_augmentation else 0,
        horizontal_flip=args.use_augmentation,
        vertical_flip=args.vertical_flip,
        brightness_range=args.brightness_range,
        use_clahe=args.use_clahe
    )

    print(f"Đang tạo bộ sinh dữ liệu từ {args.data_dir}...")
    if args.use_class_weights:
        train_generator, val_generator, class_weights = data_preprocessor.create_data_generators(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            compute_weights=args.use_class_weights
        )
    else:
        train_generator, val_generator = data_preprocessor.create_data_generators(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            compute_weights=args.use_class_weights
        )
        class_weights = None
    
    # Lấy danh sách tên lớp từ generator
    class_names = list(train_generator.class_indices.keys())

    num_classes = len(class_names)
    print(f"Đã tìm thấy {num_classes} lớp: {class_names}")

    # Vẽ biểu đồ phân phối lớp
    data_preprocessor._plot_class_distribution(args.data_dir, train_generator.class_indices)

    # Tạo mô hình
    print(f"Khởi tạo mô hình {args.model_type}...")
    bird_classifier = BirdClassifier(num_classes=num_classes, img_size=args.img_size)

    if args.model_type == 'custom_cnn':
        model = bird_classifier.create_custom_cnn()
    elif args.model_type == 'resnet50':
        model = bird_classifier.create_resnet50_model(fine_tune_layers=args.fine_tune_layers)
    elif args.model_type == 'mobilenetv2':
        model = bird_classifier.create_mobilenetv2_model()
    else:
        raise ValueError(f"Loại mô hình không hợp lệ: {args.model_type}")

    model = bird_classifier.compile_model(model)
    model.summary()

    # Huấn luyện mô hình
    print(f"Bắt đầu huấn luyện với {args.epochs} epochs...")
    history = bird_classifier.train_model(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=args.epochs,
        initial_learning_rate=args.learning_rate,
        min_learning_rate=args.learning_rate / 100
    )

    plot_training_history(history)

    # Đánh giá mô hình
    print("Đánh giá mô hình trên tập validation...")
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Dự đoán để tạo confusion matrix
    print("Tạo confusion matrix...")
    val_steps = val_generator.samples // val_generator.batch_size + 1
    y_pred = []
    y_true = []

    for i in range(val_steps):
        try:
            x_batch, y_batch = next(val_generator)
            pred_batch = model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(pred_batch, axis=1))
            y_true.extend(np.argmax(y_batch, axis=1))
        except StopIteration:
            break

    plot_confusion_matrix(y_true, y_pred, class_names)
    print_classification_report(y_true, y_pred, class_names)

    # Lưu mô hình
    print(f"Lưu mô hình với tên {args.model_name}...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_manager = ModelManager(base_dir)

    # Chuyển đổi history sang dict
    history_dict = {key: [float(val) for val in history.history[key]] for key in history.history}

    model_path = model_manager.save_model(
        model=model,
        model_name=args.model_name,
        training_history=history_dict,
        class_names=class_names,
        img_size=args.img_size,
        use_clahe=args.use_clahe
    )

    print(f"Mô hình đã được lưu tại: {model_path}")
    print("✅ Quá trình huấn luyện hoàn tất!")

if __name__ == '__main__':
    main()