import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import os

class BirdClassifier:
    def __init__(self, num_classes, img_size=224):
        self.num_classes = num_classes
        self.img_size = img_size
        
    def create_custom_cnn(self):
        """Tạo mô hình CNN từ đầu với regularization"""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         kernel_regularizer=l2(0.01),
                         input_shape=(self.img_size, self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=l2(0.01)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=l2(0.01)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=l2(0.01)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_resnet50_model(self, fine_tune_layers=30):
        """Tạo mô hình dựa trên ResNet50 với fine-tuning"""
        # Load ResNet50 pre-trained model
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Đóng băng các layer trừ n layer cuối
        if fine_tune_layers > 0:
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
            for layer in base_model.layers[-fine_tune_layers:]:
                layer.trainable = True
        else:
            base_model.trainable = False
        
        # Thêm các layer mới với regularization
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(1024, activation='relu',
                        kernel_regularizer=l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def create_mobilenetv2_model(self, fine_tune_layers=30):
        """Tạo mô hình dựa trên MobileNetV2 với fine-tuning"""
        # Load MobileNetV2 pre-trained model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Đóng băng các layer trừ n layer cuối
        if fine_tune_layers > 0:
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
            for layer in base_model.layers[-fine_tune_layers:]:
                layer.trainable = True
        else:
            base_model.trainable = False
        
        # Thêm các layer mới với regularization
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(1024, activation='relu',
                        kernel_regularizer=l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def compile_model(self, model):
        """Biên dịch mô hình với optimizer và loss function phù hợp"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Tăng learning rate
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model(self, model, train_generator, val_generator, epochs=100,
                    initial_learning_rate=0.001, min_learning_rate=0.00001):
        """Huấn luyện mô hình với callbacks"""
        # Tạo đường dẫn đến thư mục models
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        callbacks = [
            # Early stopping để tránh overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            # Giảm learning rate khi model không cải thiện
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=min_learning_rate,
                verbose=1
            ),
            # Lưu model tốt nhất
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(models_dir, 'best_model.h5'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history 