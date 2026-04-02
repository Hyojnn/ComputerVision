import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import os

# 현재 스크립트의 디렉토리를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')

# results 폴더 확인
os.makedirs(results_dir, exist_ok=True)

# 1. MNIST 데이터셋 로드
print("MNIST 데이터셋 로드 중...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리 (0~1 정규화)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. 모델 구축
model = Sequential([
    Flatten(input_shape=(28, 28)),      # 28x28 2차원 배열을 1차원 배열로 평탄화
    Dense(128, activation='relu'),      # 128개의 노드를 가진 은닉층
    Dense(10, activation='softmax')     # 10개의 클래스(0~9)로 출력
])

# 4. 모델 훈련 설정 (컴파일)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
print("모델 훈련 시작...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 6. 모델 평가
print("\n모델 평가:")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 정확도: {test_acc:.4f}")

# 7. 결과 시각화 및 저장
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '과제1_결과.png'))
print("결과 그래프가 '5주차_과제/results/과제1_결과.png'에 저장되었습니다.")
