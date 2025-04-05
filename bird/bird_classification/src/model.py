import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import os
import numpy as np

class BirdClassifier:
    def __init__(self, num_classes, img_size=224):
        self.num_classes = num_classes
        self.img_size = img_size
        
    def create_custom_cnn(self):
        """Tạo mô hình CNN từ đầu với regularization mạnh hơn"""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         kernel_regularizer=l2(0.02),
                         input_shape=(self.img_size, self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=l2(0.02)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=l2(0.02)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=l2(0.02)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=l2(0.02)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu',
                        kernel_regularizer=l2(0.02)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_resnet50_model(self, fine_tune_layers=30):
        """Tạo mô hình dựa trên ResNet50 với fine-tuning và regularization mạnh hơn"""
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
        
        # Thêm các layer mới với regularization mạnh hơn
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(1024, activation='relu',
                        kernel_regularizer=l2(0.02))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=l2(0.02))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=l2(0.02))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def create_mobilenetv2_model(self, fine_tune_layers=30):
        """Tạo mô hình dựa trên MobileNetV2 với fine-tuning và regularization mạnh hơn"""
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
        
        # Thêm các layer mới với regularization mạnh hơn
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(1024, activation='relu',
                        kernel_regularizer=l2(0.02))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=l2(0.02))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=l2(0.02))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def compile_model(self, model):
        """Biên dịch mô hình với optimizer và learning rate schedule phù hợp"""
        # Sử dụng cosine decay learning rate với warmup
        initial_learning_rate = 0.001
        decay_steps = 1000
        warmup_steps = 100
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            decay_steps,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
        
        # Thêm warmup
        lr_schedule = tf.keras.optimizers.schedules.WarmUp(
            lr_schedule,
            warmup_steps,
            initial_learning_rate
        )
        
        # Sử dụng Adam với gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0  # Gradient clipping
        )
        
        # Sử dụng label smoothing để giảm overfitting
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        return model
    
    def mixup_data(self, x, y, alpha=0.2):
        """Áp dụng mixup augmentation cho dữ liệu
        
        Args:
            x: Dữ liệu đầu vào
            y: Nhãn
            alpha: Tham số cho phân phối Beta
            
        Returns:
            Dữ liệu đã được mixup
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = tf.shape(x)[0]
        index = tf.random.shuffle(tf.range(batch_size))

        mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
        mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
        
        return mixed_x, mixed_y
    
    def train_model(self, model, train_generator, val_generator, epochs=100,
                    initial_learning_rate=0.001, min_learning_rate=0.00001,
                    use_mixup=True, mixup_alpha=0.2):
        """Huấn luyện mô hình với callbacks cải tiến và mixup augmentation"""
        # Tạo đường dẫn đến thư mục models
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Tạo custom training loop với mixup
        if use_mixup:
            # Khởi tạo biến theo dõi
            best_val_acc = 0.0
            prev_val_loss = float('inf')
            patience_counter = 0
            
            # Khởi tạo history
            history = {
                'loss': [],
                'accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            # Tạo custom training step
            @tf.function
            def train_step(x, y):
                with tf.GradientTape() as tape:
                    # Áp dụng mixup
                    mixed_x, mixed_y = self.mixup_data(x, y, alpha=mixup_alpha)
                    # Dự đoán
                    predictions = model(mixed_x, training=True)
                    # Tính loss
                    loss = tf.keras.losses.categorical_crossentropy(mixed_y, predictions)
                    loss = tf.reduce_mean(loss)
                
                # Tính gradient và áp dụng gradient clipping
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer = model.optimizer
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                return loss, predictions
            
            # Tạo custom training loop
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                
                # Training
                train_loss = 0
                train_acc = 0
                train_steps = 0
                
                for x_batch, y_batch in train_generator:
                    loss, predictions = train_step(x_batch, y_batch)
                    train_loss += loss
                    train_acc += tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_batch, predictions))
                    train_steps += 1
                
                train_loss /= train_steps
                train_acc /= train_steps
                
                # Validation
                val_loss = 0
                val_acc = 0
                val_steps = 0
                
                for x_batch, y_batch in val_generator:
                    predictions = model(x_batch, training=False)
                    loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)
                    val_loss += tf.reduce_mean(loss)
                    val_acc += tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_batch, predictions))
                    val_steps += 1
                
                val_loss /= val_steps
                val_acc /= val_steps
                
                # Lưu metrics vào history
                history['loss'].append(float(train_loss))
                history['accuracy'].append(float(train_acc))
                history['val_loss'].append(float(val_loss))
                history['val_accuracy'].append(float(val_acc))
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Early stopping check
                if epoch > 0 and val_loss > prev_val_loss:
                    patience_counter += 1
                    if patience_counter >= 15:  # Early stopping patience
                        print("Early stopping triggered")
                        break
                else:
                    patience_counter = 0
                
                prev_val_loss = val_loss
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model.save(os.path.join(models_dir, 'best_model.h5'))
                    print("Saved best model")
            
            # Tạo history object để tương thích với code cũ
            history_obj = type('History', (), {'history': history})()
            
            return history_obj
        
        else:
            # Sử dụng fit thông thường nếu không dùng mixup
            callbacks = [
                # Early stopping với patience cao hơn
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    min_delta=0.001
                ),
                # ReduceLROnPlateau với patience cao hơn
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
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
                ),
                # Thêm TensorBoard callback
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(models_dir, 'logs'),
                    histogram_freq=1,
                    write_graph=True,
                    update_freq='epoch'
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