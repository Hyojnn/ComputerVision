import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# 현재 실행 중인 파일의 절대 경로를 바탕으로 폴더 지정
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "images/edgeDetectionImage.jpg")

# 이미지 로드
img = cv.imread(image_path)
if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()

# Matplotlib 출력을 위한 BGR에서 RGB로 색상 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 1. 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Sobel 필터를 사용하여 x축과 y축 방향의 에지를 검출
# CV_64F를 사용하여 연산 중 발생할 수 있는 데이터 손실 방지, ksize는 3으로 설정
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 3. 에지 강도 계산
magnitude = cv.magnitude(sobel_x, sobel_y)

# 4. 에지 강도 이미지를 시각화하기 위해 uint8로 변환
magnitude = cv.convertScaleAbs(magnitude)

# 5. 결과 시각화 (원본 이미지와 에지 강도 이미지 나란히 출력)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude, cmap='gray')
plt.title('Edge Magnitude')
plt.axis('off')

plt.tight_layout()
plt.savefig('과제1_결과.png', dpi=300, bbox_inches='tight')
plt.close()
