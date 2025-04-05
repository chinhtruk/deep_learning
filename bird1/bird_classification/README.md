# Hệ Thống Nhận Diện Loài Chim Sử Dụng Deep Learning 🦜

Dự án này sử dụng các kỹ thuật học sâu (Deep Learning) để phân loại các loài chim dựa trên hình ảnh, với nhiều mô hình khác nhau như CNN tùy chỉnh, ResNet50 và MobileNetV2.

## Tính Năng Chính

- **Đa dạng mô hình**: Hỗ trợ nhiều kiến trúc mô hình (CNN tùy chỉnh, ResNet50, MobileNetV2)
- **Tiền xử lý dữ liệu nâng cao**: Tăng cường dữ liệu, cân bằng lớp, cải thiện độ tương phản với CLAHE
- **Quản lý mô hình**: Lưu và tải mô hình cùng với metadata (lịch sử huấn luyện, tên lớp, tham số tiền xử lý)
- **Đánh giá toàn diện**: Vẽ đồ thị quá trình huấn luyện, confusion matrix, báo cáo phân loại
- **Giao diện web**: Ứng dụng Flask để tải lên và dự đoán ảnh
- **Tham số linh hoạt**: Dễ dàng điều chỉnh các tham số huấn luyện qua dòng lệnh

## Cấu Trúc Dự Án

```
bird_classification/
├── data/
│   ├── train/                # Dữ liệu huấn luyện
│   ├── test/                 # Dữ liệu kiểm tra
│   └── images to predict/    # Thư mục chứa ảnh tạm thời để dự đoán
├── models/                   # Lưu trữ các model đã huấn luyện và metadata
├── src/
│   ├── app.py                # Ứng dụng web Flask
│   ├── data_preprocessing.py # Xử lý và tăng cường dữ liệu
│   ├── image_utils.py        # Xử lý ảnh và cải thiện chất lượng
│   ├── main.py               # Script huấn luyện chính
│   ├── model.py              # Định nghĩa các kiến trúc mô hình
│   ├── model_manager.py      # Quản lý lưu và tải mô hình
│   └── utils.py              # Các hàm tiện ích và đánh giá
└── requirements.txt          # Các thư viện cần thiết
```

## Cài Đặt

1. Tạo môi trường ảo (virtual environment):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Chuẩn Bị Dữ Liệu

1. Tải dataset chim (như CUB-200-2011, NABirds hoặc dataset tùy chỉnh)
2. Tổ chức dữ liệu theo cấu trúc:
```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
└── test/
    ├── class1/
    │   └── image1.jpg
    └── class2/
        └── image1.jpg
```

## Huấn Luyện Mô Hình

Chạy script huấn luyện với các tham số tùy chỉnh:

```bash
python src/main.py --data_dir data/train --model_type mobilenetv2 --img_size 224 --batch_size 32 --epochs 30
```

Các tham số có thể điều chỉnh:

- `--data_dir`: Thư mục chứa dữ liệu huấn luyện
- `--model_type`: Loại mô hình (custom_cnn, resnet50, mobilenetv2)
- `--model_name`: Tên mô hình để lưu
- `--img_size`: Kích thước ảnh đầu vào
- `--batch_size`: Kích thước batch
- `--epochs`: Số epochs huấn luyện
- `--learning_rate`: Tốc độ học
- `--fine_tune_layers`: Số lớp cuối cùng để fine-tune
- `--validation_split`: Tỷ lệ dữ liệu validation
- `--use_class_weights`: Sử dụng trọng số lớp
- `--use_clahe`: Sử dụng CLAHE để cải thiện độ tương phản

## Chạy Ứng Dụng Web

Khởi động ứng dụng web để dự đoán ảnh:

```bash
python src/app.py
```

Sau đó truy cập http://localhost:5000 trong trình duyệt để tải lên và dự đoán ảnh chim.

## Kết Quả

- Mô hình được lưu trong thư mục `models/` cùng với metadata
- Đồ thị quá trình huấn luyện được hiển thị và lưu
- Confusion matrix và báo cáo phân loại chi tiết
- Biểu đồ phân phối lớp để phân tích dữ liệu

## Đánh Giá Mô Hình

Mô hình được đánh giá dựa trên:
- Độ chính xác (Accuracy)
- Precision, Recall và F1-score
- Confusion matrix
- Đồ thị loss và accuracy trong quá trình huấn luyện

## Yêu Cầu Hệ Thống

- Python 3.7+
- TensorFlow 2.x
- GPU (khuyến nghị) hoặc CPU
- RAM: 8GB+
- Ổ cứng: 10GB+ (tùy thuộc kích thước dataset)

## Tính Năng Nâng Cao

- **Xử lý ảnh nâng cao**: Sử dụng CLAHE để cải thiện độ tương phản
- **Transfer Learning**: Tận dụng các mô hình đã được huấn luyện trước trên ImageNet
- **Fine-tuning**: Điều chỉnh số lớp được fine-tune trong các mô hình transfer learning
- **Callbacks thông minh**: Early stopping, ReduceLROnPlateau, ModelCheckpoint
- **Lưu trữ metadata**: Lưu thông tin về kích thước ảnh và cài đặt CLAHE cùng với mô hình
