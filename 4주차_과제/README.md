# 📷 OpenCV 4주차 과제 정리

본 저장소는 컴퓨터비전 OpenCV 4주차 과제(1~3)를 수행한 결과를 담고 있습니다.

---

## 📌 과제 1: SIFT를 이용한 특징점 검출 및 시각화
`01_sift_detection.py`
주어진 이미지(`mot_color70.jpg`)를 이용하여 SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용해 특징점을 검출하고, 이를 시각화하여 원본과 함께 나란히 출력하는 과제입니다.

### 📝 전체 코드
```python
import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 로드
img = cv.imread('mot_color70.jpg')
if img is None:
    print('Failed to load image.')
    exit()

# BGR을 RGB로 변환 (matplotlib 시각화를 위해)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 2. SIFT 객체 생성
# nfeatures를 지정하여 너무 많은 특징점이 검출되지 않도록 조절 (예: 500개)
sift = cv.SIFT_create(nfeatures=500)

# 3. 특징점 검출 및 디스크립터 계산
# 특징점(keypoints)과 디스크립터(descriptors) 반환
keypoints, descriptors = sift.detectAndCompute(img, None)

# 4. 특징점 시각화
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS를 사용하면 특징점의 크기와 방향도 원으로 표시됨
img_with_keypoints = cv.drawKeypoints(
    img_rgb, 
    keypoints, 
    None, 
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 5. matplotlib을 이용한 결과 출력
plt.figure(figsize=(12, 6))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 특징점이 표시된 이미지
plt.subplot(1, 2, 2)
plt.imshow(img_with_keypoints)
plt.title('SIFT Keypoints')
plt.axis('off')

plt.tight_layout()
plt.savefig('과제1_결과.png')
# plt.show()
```

### 🔑 주요 코드 및 설명
```python
sift = cv.SIFT_create(nfeatures=500)
keypoints, descriptors = sift.detectAndCompute(img, None)
img_with_keypoints = cv.drawKeypoints(
    img_rgb, 
    keypoints, 
    None, 
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
```
* **`cv.SIFT_create`**: SIFT 알고리즘에 기반한 특징점 검출기 객체를 만듭니다. 결과가 너무 복잡하게 출력되지 않도록 `nfeatures` 옵션을 주어 추출할 특징점의 최대 개수를 제한할 수 있습니다.
* **`detectAndCompute`**: 이미지 크기(Scale)와 회전(Rotation)의 변화에 강인한 특징점의 위치 정보(keypoints)와 다른 점들과 구분할 수 있는 서술자 벡터(descriptors)를 계산합니다.
* **`cv.drawKeypoints`**: 검출된 keypoint들을 원본 이미지 위에 그려 시각화합니다. `DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` 옵션을 켜면 각 특징점마다 방향(Orientation)과 스케일 크기 정보를 포함한 예쁜 원 형태로 표현할 수 있습니다.

### 🖥 실행 결과 화면
![과제1 결과](./과제1_결과.png)

---

## 📌 과제 2: SIFT를 이용한 두 영상 간 특징점 매칭
`02_sift_matching.py`
두 개의 다른 시점에 찍힌 이미지(`mot_color70.jpg`, `mot_color83.jpg`)를 불러온 후, SIFT 알고리즘으로 추출한 특징점들을 비교(Matching)하여 어울리는 대응점들을 찾아 시각화하는 과제입니다.

### 📝 전체 코드
```python
import cv2 as cv
import matplotlib.pyplot as plt

# 1. 두 이미지 로드
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color83.jpg') 

if img1 is None or img2 is None:
    print('Failed to load images.')
    exit()

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# 2. SIFT 객체 생성
sift = cv.SIFT_create()

# 3. 특징점 및 디스크립터 추출
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 4. 특징점 매칭 (BFMatcher 적용)
# SIFT는 L2 노름을 사용하므로 cv.NORM_L2 지정
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# 좋은 매칭점만 선별 (최근접 이웃 거리 비율 테스트)
good_matches = []
ratio_thresh = 0.7  # 임계값 0.7
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# 5. 매칭 결과 시각화
img_matches = cv.drawMatches(
    img1_rgb, kp1, 
    img2_rgb, kp2, 
    good_matches, None, 
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 6. matplotlib 결과 출력
plt.figure(figsize=(15, 6))
plt.imshow(img_matches)
plt.title('SIFT Matching (Good Matches)')
plt.axis('off')

plt.savefig('과제2_결과.png')
# plt.show()
```

### 🔑 주요 코드 및 설명
```python
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```
* **`cv.BFMatcher`**: Brute-Force Matcher 방식입니다. 하나의 특징점과 대응될 가능성이 있는 모든 다른 이미지 내의 특징점들과 계산을 통해 물리적 거리가 가까운 것(유사도가 높은 것)을 찾습니다.
* **`knnMatch(k=2)`**: 각 특징점마다 거리가 가장 가까운 k개(여기서는 2개)의 좋은 후보를 매칭점 배열로 반환합니다.
* **비율 테스트(Ratio test)**: 거리와 유사도 검증을 위해 1등 매칭(m)과 2등 매칭(n)의 거리를 비교하는데, 1등이 2등보다 확실하게 더 가깝고 구분이 될 때만 임계값(`0.7`)에 의해 통과되도록 하여 우수한(Good) 매칭점만 필터링합니다.

### 🖥 실행 결과 화면
![과제2_결과](./과제2_결과.png)

---

## 📌 과제 3: 호모그래피를 이용한 이미지 정합 (Image Alignment)
`03_homography_alignment.py`
두 샘플 이미지(`img1.jpg`, `img2.jpg`)에서 찾아낸 좋은 특징점 매칭 정보(`good_matches`)를 바탕으로, 카메라 시점 변화 정보인 호모그래피 매트릭스를 계산하고 한쪽 이미지를 원근 변환(Perspective Warping)시켜 넓은 시야의 파노라마 영상을 만들어내는 과제입니다.

### 📝 전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 두 이미지 로드
img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')

if img1 is None or img2 is None:
    print('Failed to load images.')
    exit()

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# 2. SIFT 객체 생성 및 특징점 검출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_rgb, None)
kp2, des2 = sift.detectAndCompute(img2_rgb, None)

# 3. 특징점 매칭 및 좋은 매칭점 선별
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

img_matches = cv.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 4. 호모그래피 행렬 계산
if len(good_matches) > 4: # 호모그래피는 최소 4점 이상의 매칭이 필요
    # 매칭점의 좌표 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # RANSAC을 이용한 호모그래피 계산
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    # 5. 한 이미지를 변환하여 정렬 (파노라마)
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    out_w, out_h = w1 + w2, max(h1, h2)
    
    # img1 변환
    warped_img1 = cv.warpPerspective(img1_rgb, H, (out_w, out_h))
    
    # 두 이미지 붙이기
    panorama = warped_img1.copy()
    panorama[0:h2, 0:w2] = img2_rgb
    
    # 6. 결과 출력
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.imshow(img_matches)
    plt.title('Matching Result')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.imshow(panorama)
    plt.title('Warped Image (Panorama)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('과제3_결과.png')
else:
    print('Not enough matches are found')
```

### 🔑 주요 코드 및 설명
```python
    # RANSAC을 이용한 호모그래피 분해 및 계산
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    # 5. 한 이미지를 변환하여 정렬 (파노라마)
    out_w, out_h = w1 + w2, max(h1, h2)
    warped_img1 = cv.warpPerspective(img1_rgb, H, (out_w, out_h))
```
* **`cv.findHomography`**: 두 평면(이미지) 사이의 투시 변환 행렬인 호모그래피 `H`를 계산합니다. 오차를 보정하기 위해 `cv.RANSAC` 알고리즘을 사용해 잘못 매칭된 Outlier 좌표들의 영향을 받지 않고 가장 올바른 변환 공식을 찾아냅니다. 
* **`cv.warpPerspective`**: 구한 호모그래피 `H` 매트릭스를 적용해 이미지를 투시 변환시킵니다. 여기서는 두 이미지를 하나로 합칠 충분한 공간(`w1+w2` 크기의 커다란 캔버스)를 `out_w, out_h`로 지정해 변형한 뒤 또 다른 기준 이미지를 합쳐 파노라마 결과를 완성하였습니다.

### 🖥 실행 결과 화면
![과제3_결과](./과제3_결과.png)
