import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import json

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

# 3~5. 모델 불러오기 또는 새로 훈련하기
model_path = os.path.join(results_dir, 'cifar10_model.keras')
history_path = os.path.join(results_dir, 'cifar10_history.json')

history_dict = None

if os.path.exists(model_path):
    # 저장된 모델이 있으면 훈련 건너뛰고 바로 불러오기
    print(f"\n저장된 모델을 발견했습니다! 훈련을 생략하고 '{model_path}'에서 불러옵니다.")
    model = tf.keras.models.load_model(model_path)
    trained_now = False
    
    # 훈련 기록(그래프 데이터)이 있으면 같이 불러오기
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history_dict = json.load(f)
else:
    print("\n저장된 모델이 없습니다. 새롭게 훈련을 시작합니다 (이후에는 자동으로 저장되어 훈련이 생략될 수 있습니다).")
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
    
    # 훈련 완료 직후 기록(history) 저장
    history_dict = history.history
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f)
        
    # 훈련 완료 후 모델 로컬 저장
    model.save(model_path)
    print(f"모델 및 훈련 기록 저장 완료! '{model_path}' 경로에 영구적으로 저장되었습니다.")
    trained_now = True

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
    if history_dict is not None:
        # 훈련 기록(그래프 데이터)이 존재하면 그래프와 함께 표출
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['accuracy'], label='Training Accuracy')
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
        plt.title('CNN Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_rgb)
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
    else:
        # 기록 파일이 유실되었거나 없는 경우 이미지만 결과로 표출
        plt.figure(figsize=(6, 5))
        plt.imshow(img_rgb)
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)\n(Loaded Pre-trained Model)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '과제2_결과.png'))
    print("결과 이미지가 '5주차_과제/results/과제2_결과.png'에 저장되었습니다.")
else:
    print(f"\n오류: 테스트 이미지를 찾을 수 없습니다: {img_path}")
