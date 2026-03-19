import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "images/coffee cup.JPG")

img = cv.imread(image_path)
if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()

# BGR에서 RGB로 색상 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 1. 초기 마스크, 배경 모델, 전경 모델 생성
# np.zeros((1, 65), np.float64)로 초기화
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 2. 초기 사각형 영역은 (x, y, width, height) 형식으로 설정
h, w = img.shape[:2]
# 커피 컵 영역을 덮을 수 있는 적절한 사각형을 수동으로 설정 (수정 가능)
rect = (w // 6, h // 6, w * 2 // 3, h * 2 // 3)

# 3. cv.grabCut()를 사용하여 대화식 분할을 수행
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

# 4. 마스크를 사용하여 이미지 복원 및 원본 이미지에서 배경 제거
# np.where()를 사용하여 마스크 값을 0(배경) 또는 1(전경)로 변경
# GC_BGD(0), GC_PR_BGD(2)는 0으로 처리, GC_FGD(1), GC_PR_FGD(3)는 1로 처리
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 원본 이미지에 마스크 이미지를 곱하여 배경을 제거
img_extracted = img * mask2[:, :, np.newaxis]
img_extracted_rgb = cv.cvtColor(img_extracted, cv.COLOR_BGR2RGB)

# 5. 결과 시각화
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask2, cmap='gray')
plt.title('Mask Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_extracted_rgb)
plt.title('Extracted Object')
plt.axis('off')

plt.tight_layout()
plt.savefig('과제3_결과.png', dpi=300, bbox_inches='tight')
plt.close()
