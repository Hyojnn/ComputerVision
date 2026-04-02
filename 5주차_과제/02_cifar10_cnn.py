import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# 현재 스크립트의 디렉토리를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')

# results 폴더 확인
os.makedirs(results_dir, exist_ok=True)

# 1. CIFAR-10 데이터셋 로드
print("CIFAR-10 데이터셋 로드 중...")
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# CIFAR-10 클래스 이름 (예측 결과 확인용)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. 데이터 전처리 (0~1 범위로 정규화)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. CNN 모델 설계 (정확도 향상 버전)
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 4. 모델 훈련 설정 (컴파일)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
print("CNN 모델 훈련 시작...")
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# 6. 모델 성능 평가
print("\n모델 평가:")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 정확도: {test_acc:.4f}")

# 7. 테스트 이미지에 대한 예측 수행 (dog.jpg)
img_path = os.path.join(current_dir, 'image', 'dog.jpg')
if os.path.exists(img_path):
    # OpenCV로 이미지 로드 (BGR -> RGB 변환)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 모델 입력 크기에 맞게 이미지 리사이즈 (32x32)
    img_resized = cv2.resize(img_rgb, (32, 32))
    
    # 정규화 및 배치 차원 추가
    img_input = np.expand_dims(img_resized / 255.0, axis=0)
    
    # 예측 수행
    predictions = model.predict(img_input)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = np.max(predictions) * 100
    
    print(f"\n예측 결과: {predicted_class} (정확도: {confidence:.2f}%)")
    
    # 8. 결과 시각화 및 저장
    plt.figure(figsize=(12, 5))
    
    # 모델 정확도 변화 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 테스트 이미지 예측 결과 패널
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '과제2_결과.png'))
    print("결과 이미지가 '5주차_과제/results/과제2_결과.png'에 저장되었습니다.")
else:
    print(f"\n오류: 테스트 이미지를 찾을 수 없습니다: {img_path}")
