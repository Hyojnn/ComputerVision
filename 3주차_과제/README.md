# 📷 OpenCV 3주차 과제 정리

본 저장소는 컴퓨터비전 OpenCV 3주차 과제(1~3)를 수행한 결과를 담고 있습니다.

---

## 📌 과제 1: 소벨 에지 검출 및 결과 시각화
`01.SobelEdge.py`  
이미지를 그레이스케일로 변환한 후, Sobel 필터를 사용하여 x축과 y축 방향의 에지를 검출하고, 이를 바탕으로 에지 강도 이미지를 계산하여 시각화하는 과제입니다.

### 📝 전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "images/edgeDetectionImage.jpg")

img = cv.imread(image_path)
if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 1. 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Sobel 필터를 사용하여 x축과 y축 방향의 에지 검출
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 3. 에지 강도 계산
magnitude = cv.magnitude(sobel_x, sobel_y)

# 4. 에지 강도 이미지를 uint8로 변환
magnitude = cv.convertScaleAbs(magnitude)

# 5. 결과 시각화
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
```

### 🔑 주요 코드 및 설명
```python
# Sobel 필터를 사용하여 x축, y축 에지 검출 (ksize=3 설정)
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 에지 강도 계산 및 uint8로 변환 설정
magnitude = cv.magnitude(sobel_x, sobel_y)
magnitude = cv.convertScaleAbs(magnitude)
```
* **`cv.Sobel`**: 소벨 마스크를 사용하여 이미지상에서 미분값을 구해 에지를 찾아냅니다. 데이터 손실을 방지하기 위해 `CV_64F` 정밀도를 사용합니다.
* **`cv.magnitude`**: x축 방향과 y축 방향 벡터들의 크기를 결합해, 전체적인 에지의 강도를 계산합니다.
* **`cv.convertScaleAbs`**: 계산된 결과를 출력과 시각화가 가능하도록 절대값을 취해 `uint8` 형식(0~255 범위의 픽셀값)으로 변환해줍니다.

### 🖥 실행 결과 화면
![과제1 결과](./과제1_결과.png)

---

## 📌 과제 2: 캐니 에지 및 허프 변환을 이용한 직선 검출
`02.CannyHough.py`  
이미지에 캐니(Canny) 에지 검출을 사용하여 에지 맵을 생성한 후, 허프 변환(Hough Transform)을 응용해 직선 성분을 찾아 원본 이미지에 그려주는 과제입니다.

### 📝 전체 코드
```python
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

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_line = img.copy()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 1. Canny 에지 검출로 에지 맵 생성 (threshold: 100, 200)
edges = cv.Canny(gray, threshold1=100, threshold2=200)

# 2. 허프 변환을 사용한 직선 검출 수행
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# 3. 검출된 직선을 원본 이미지에서 붉은색으로 그림 
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img_line, (x1, y1), (x2, y2), (0, 0, 255), 2)

img_line_rgb = cv.cvtColor(img_line, cv.COLOR_BGR2RGB)

# 4. 결과 시각화
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
```

### 🔑 주요 코드 및 설명
```python
# Canny 에지 검출 (threshold1, threshold2 파라미터 적용)
edges = cv.Canny(gray, threshold1=100, threshold2=200)

# 허프 변환을 통한 직선 성분 추출
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
```
* **`cv.Canny`**: 윤곽선을 깔끔하게 한 픽셀의 두께로 검출해 내는 함수입니다. 임계값(Threshold1, 2) 설정을 통해 확실한 강력한 경계만 추출하도록 조정합니다.
* **`cv.HoughLinesP`**: 확률적 허프 변환 알고리즘으로 윤곽선 안에서 직선 성분을 빠르게 찾아 좌표를 도출합니다. `minLineLength`(최소 선 길이)과 `maxLineGap`(끊어진 선 허용 오차) 등을 적절하게 지정해서 좋은 품질의 직선들을 뽑아냅니다.
* **`cv.line`**: 검출된 좌표를 따라 영상 위에 빨간색(`(0, 0, 255)`)으로 두께가 2인 선을 실제로 그려주는 기능입니다.

### 🖥 실행 결과 화면
![과제2 결과](./과제2_결과.png)

---

## 📌 과제 3: GrabCut을 이용한 대화식 영역 분할 및 객체 추출
`03.GrabCut.py`  
사진 속에서 사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용해 불필요한 배경과 타겟 객체를 분리(Segmentation)하여 주요 피사체만 깔끔하게 추출해 내는 과제입니다.

### 📝 전체 코드
```python
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

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 1. 초기 마스크 및 배경, 전경 모델 세팅
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 2. 초기 사각형 영역 설정 (x, y, width, height)
h, w = img.shape[:2]
rect = (w // 6, h // 6, w * 2 // 3, h * 2 // 3)

# 3. 대화식 분할 알고리즘 수행 (GrabCut)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

# 4. 마스크를 사용한 원본 이미지 배경 제거 추출 작업
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
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
```

### 🔑 주요 코드 및 설명
```python
# GrabCut 알고리즘 적용 (반복 횟수 5회)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

# 추출된 mask 내부의 노이즈 정보 보정 분할
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
img_extracted = img * mask2[:, :, np.newaxis]
```
* **`cv.grabCut`**: 사각형(`rect`) 안에서 색상 및 질감 분포를 통계적으로 학습시켜 객체(전경)와 배경을 분리하는 모델 도출 알고리즘입니다. 내부적으로 가우스 혼합 모델(GMM) 파라미터들을 `bgdModel`, `fgdModel`로 학습하며 저장합니다.
* **`np.where`**: 반환된 1차 마스크 영역 안에는 확실한 배경, 아마도 배경, 확실한 전경, 아마도 전경 값이 섞여있습니다. 해당 함수를 써서 배경으로 지목된 부분들은 0, 전경은 1이라는 두 가지 이진(Binary) 마스크 값으로 확정 짓습니다.
* **`img * mask2`**: 완성된 0과 1로 된 흑백 마스크 평면 공간을 3차원(`np.newaxis`)으로 늘려 곱셈하면, 원본 이미지에서 0인 뒷배경은 모두 검은색(삭제)으로 변하고 전경 객체만 살아남게 됩니다.

### 🖥 실행 결과 화면
![과제3 추출결과](./과제3_결과.png)
