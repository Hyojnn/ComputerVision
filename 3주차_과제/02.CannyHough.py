import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "images/dabo.jpg")

img = cv.imread(image_path)
if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()

# BGR에서 RGB로 색상 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# 직선을 표시할 이미지 미리 복사
img_line = img.copy()

# 1. 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Canny 에지 검출을 사용하여 에지 맵 생성 
# threshold1, threshold2는 각각 100과 200으로 설정
edges = cv.Canny(gray, threshold1=100, threshold2=200)

# 3. 허프 변환을 사용하여 이미지에서 직선 검출
# rho, theta, threshold, minLineLength, maxLineGap 조절
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# 4. 검출된 직선을 원본 이미지에서 빨간색으로 표시
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # (0, 0, 255)는 BGR에서 빨간색, 두께는 2로 설정
        cv.line(img_line, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 결과 출력을 위해 BGR에서 RGB로 색상 변환
img_line_rgb = cv.cvtColor(img_line, cv.COLOR_BGR2RGB)

# 5. 결과 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_line_rgb)
plt.title('Detected Lines')
plt.axis('off')

plt.tight_layout()
plt.savefig('과제2_결과.png', dpi=300, bbox_inches='tight')
plt.close()
