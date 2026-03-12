# 📷 OpenCV 2주차 과제 정리

본 저장소는 컴퓨터비전 OpenCV 2주차 과제(1~3)를 수행한 결과를 담고 있습니다.

---

## 📌 과제 1: 체크보드 기반 카메라 캘리브레이션
`01.Calibration.py`  
다양한 각도에서 촬영된 체크보드(흑백 격자) 이미지들의 코너 좌표를 추출하여, 이를 기반으로 카메라의 내부 파라미터(Camera Matrix)와 왜곡 계수(Distortion Coefficients)를 계산하고 이미지 왜곡을 보정하는 과제입니다.

### 📝 전체 코드
```python
import cv2
import numpy as np
import glob
import os

# 체크보드 내부 코너 개수 (가로, 세로)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표(3D World Points) 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표 리스트
objpoints = [] # 실제 세계의 3D 점들 
imgpoints = [] # 이미지 평면의 2D 점들 

# 현재 실행 중인 파일의 디렉토리를 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
image_pattern = os.path.join(base_dir, "images/calibration_images/left*.jpg")
# 이미지를 정렬하여 순서대로 처리되도록 개선
images = sorted(glob.glob(image_pattern))

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img_size is None:
        img_size = gray.shape[::-1]

    # 이미지에서 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너가 발견되면 좌표 저장 (코너 검출 실패 이미지는 제외됨)
    if ret == True:
        objpoints.append(objp)
        
        # 코너 좌표 정밀화 (Sub-pixel accuracy)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 코너 그리기 시각화
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Checking Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 카메라 행렬 K와 왜곡 계수 dist 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K) 

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
sample_img = cv2.imread(images[0])
# 왜곡 보정 적용
dst = cv2.undistort(sample_img, K, dist, None, K)

# 결과 비교 시각화
cv2.imshow('Original Image', sample_img)
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 🔑 주요 코드 및 설명
```python
    # 이미지에서 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret == True:
        ...
        # 카메라 행렬 K와 왜곡 계수 dist 계산
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        ...
        # 왜곡 보정 적용
        dst = cv2.undistort(sample_img, K, dist, None, K)
```
* **`cv2.findChessboardCorners`**: 이미지에서 흑백 격자 무늬인 체크보드의 코너들을 자동으로 찾아주는 함수입니다. (2D 좌표 검출 기능)
* **`cv2.calibrateCamera`**: 확보한 각각의 2D와 3D 대응 좌표를 사용하여 카메라 내부 행렬 및 렌즈에서 발생하는 비선형 굴절률(왜곡 계수)를 도출해냅니다.
* **`cv2.undistort`**: 도출해낸 원근 및 굴절 왜곡 행렬을 반대로 적용하여, 원본 이미지의 휜 부분을 직선 형태로 바르게 펴서 정상 스케일로 시각화해 줍니다.

🖥 실행 결과 화면

<img width="466" height="99" alt="스크린샷 2026-03-12 오후 3 27 30" src="https://github.com/user-attachments/assets/d91658d4-c94c-402b-9f28-0b983f03e1b7" />
<img width="1273" height="494" alt="스크린샷 2026-03-12 오후 3 29 09 1" src="https://github.com/user-attachments/assets/ce60ccb6-4ea4-474e-bbe9-a6d543de6d9c" />

---

## 📌 과제 2: 이미지 Rotation & Transformation
`02.Transformation.py`  
주어진 하나의 이미지에 회전(+30도), 스케일링(0.8배), 평행이동(+80x, -40x) 아핀 변환(Affine Transformation) 파라미터 행렬을 수학적으로 직접 연산하여 한 번에 합쳐 적용해 보는 과제입니다.

### 📝 전체 코드
```python
import cv2
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "images/transformation_images/soccer.jpg")

# 1. 이미지 로드
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()

# 원본 이미지 크기 및 기준점 정의
h, w = img.shape[:2]
center = (w // 2, h // 2)

# -----------------------------
# 2. 회전 및 크기 조절 (Rotation & Scaling)
# -----------------------------
# 중심을 기준으로 30도 회전, 크기는 0.8배로 조절
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D(center, angle, scale)

# -----------------------------
# 3. 평행 이동 (Translation)
# -----------------------------
# x축 방향으로 +80px, y축 방향으로 -40px 만큼 평행 이동
tx = 80
ty = -40

# 회전 행렬의 3번째 열(평행 이동 성분)에 tx, ty 더하기
M[0, 2] += tx
M[1, 2] += ty

# -----------------------------
# 4. 아핀 변환 적용 (Affine Transformation)
# -----------------------------
# 변환 행렬 M을 이미지에 적용
dst = cv2.warpAffine(img, M, (w, h))

# -----------------------------
# 5. 결과 시각화
# -----------------------------
cv2.imshow('Original Image', img)
cv2.imshow('Transformed Image', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 🔑 주요 코드 및 설명
```python
# 중심을 기준으로 30도 회전, 크기는 0.8배로 조절
M = cv2.getRotationMatrix2D(center, angle, scale)

# 회전 행렬의 3번째 열(평행 이동 성분)에 tx, ty 더하기
M[0, 2] += tx
M[1, 2] += ty

# 변환 행렬 M을 이미지에 적용
dst = cv2.warpAffine(img, M, (w, h))
```
* **`cv2.getRotationMatrix2D`**: 기준 좌표로부터 원하는 회전과 비율을 넣으면 그 수식에 맞는 2x3 아핀 행렬 수식을 자동으로 세팅해줍니다.
* **`M[0, 2] += tx; M[1,2] += ty`**: 반환된 2x3 행렬의 가장 오른쪽 열은 변환의 x축, y축 평행 이동(오프셋)을 담당합니다. 이곳에 이동 값을 수동으로 더해주어 하나의 행렬로 회전/크기/이동을 조합할 수 있습니다.
* **`cv2.warpAffine`**: 위에서 조합 완성된 2x3 기하학적 아핀 변환 행렬 공식을 원본 픽셀 전체에 일괄 적용(매핑)합니다.

🖥 실행 결과 화면

<img width="1890" height="638" alt="스크린샷 2026-03-12 오후 3 30 54" src="https://github.com/user-attachments/assets/2f5a52fa-5cc7-43c0-ac52-2852df23ab11" />

---

## 📌 과제 3: Stereo Disparity 기반 Depth 추정
`03.Depth.py`  
서로 다른 위치에서 동일 타이밍에 찍힌 Left/Right 두 스테레오 카메라 렌즈 사진 간의 픽셀 편차(Disparity)를 블록 단위로 매칭하여 측정하고, 그 편차를 이용해 카메라와 물체 사이의 거리(Depth)를 구하고 색깔 지도로 시각화해보는 과제입니다.

### 📝 전체 코드
```python
import cv2
import numpy as np
import os
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 이미지를 로드하고 그레이스케일로 변환
base_dir = Path(__file__).parent.resolve()
left_color = cv2.imread(str(base_dir / "images/left.png"))
right_color = cv2.imread(str(base_dir / "images/right.png"))

gray_left = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# 카메라 파라미터 제공
f = 700.0  # 초점 거리
B = 0.12   # 베이스라인

# ROI 설정 박스 위치들
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# -----------------------------
# 2. Disparity(시차) 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity_int = stereo.compute(gray_left, gray_right)
disparity = disparity_int.astype(np.float32) / 16.0

# -----------------------------
# 3. Depth(깊이) 계산 (Z = fB / d)
# -----------------------------
valid_mask = disparity > 0
depth_map = np.zeros_like(disparity)
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 4. ROI별 편차 및 거리 분석 결과 정보 출력
# -----------------------------
results = {}
for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    roi_valid = valid_mask[y:y+h, x:x+w]
    
    if np.any(roi_valid):
        avg_disp = np.mean(roi_disp[roi_valid])
        avg_depth = np.mean(roi_depth[roi_valid])
    else:
        avg_disp, avg_depth = 0, 0
        
    results[name] = {"avg_disp": avg_disp, "avg_depth": avg_depth}

print(f"{'ROI':<10} | {'Avg Disparity':<15} | {'Avg Depth':<10}")
for name, data in results.items():
    print(f"{name:<10} | {data['avg_disp']:<15.4f} | {data['avg_depth']:<10.4f}")

closest_roi = min(results.items(), key=lambda item: item[1]['avg_depth'])
farthest_roi = max(results.items(), key=lambda item: item[1]['avg_depth'])

print(f"\n가장 가까운 객체: {closest_roi[0]} (Avg Depth: {closest_roi[1]['avg_depth']:.4f})")
print(f"가장 먼 객체: {farthest_roi[0]} (Avg Depth: {farthest_roi[1]['avg_depth']:.4f})")

... # 이후 시각화 정규화 코드 및 cv2.imshow 부분 생략 (생략된 부분은 리포지토리 파일 참고)
```

### 🔑 주요 코드 및 설명
```python
# Disparity 계산. 결과는 가중치 16배 정수치.
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity_int = stereo.compute(gray_left, gray_right)
disparity = disparity_int.astype(np.float32) / 16.0

# 수식: Z = fB / d (물체의 깊이=거리)
depth_map[valid_mask] = (f * B) / disparity[valid_mask]
```
* **`cv2.StereoBM_create`**: Block Matching 방식으로 한 이미지의 블록을 기준으로 다른 이미지 좌우 픽셀을 순회하여 매칭 시차를 찾아내는 객체를 반환합니다.
* **`.astype(np.float32) / 16.0`**: BM 방식 모델이 속도를 위해 16가중치의 정수로 결과값을 계산하여 뱉어주므로, 실계산용으로 변환하기 위해 실수 자료형으로 바꾸고 16으로 나누어 소수점 자리수를 회복하는 부분입니다.
* **`Z = (f * B) / d`**: 삼각측량의 법칙에 따른 물리적인 거리 지표 공식입니다. Disparity(`d`)가 크면 클수록 두 카메라 이미지 상 편차가 강렬하다는 뜻이므로 렌즈에 가까운 것이며, 따라서 Depth(`Z`) 값은 짧게 나옵니다.

🖥 실행 결과 화면

<img width="891" height="390" alt="스크린샷 2026-03-12 오후 3 32 22" src="https://github.com/user-attachments/assets/cdcf7432-8dd3-4287-8751-b050a11e1474" />

