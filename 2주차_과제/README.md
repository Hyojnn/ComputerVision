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

# 체크보드 내부 코너 개수 (가로 9, 세로 6)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 세계 크기 (단위: mm)
square_size = 25.0

# 서브픽셀(Sub-pixel) 정밀도를 위한 반복 종료 조건 설정
# TERM_CRITERIA_EPS: 지정된 정확도에 도달하면 종료
# TERM_CRITERIA_MAX_ITER: 지정된 최대 반복 횟수(30번)에 도달하면 종료
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 세계의 3D 좌표(World Points) 생성 (Z=0으로 가정)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# np.mgrid를 사용하여 그리드 형태의 좌표 생성 후 전치 및 형태 변경
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# 각 좌표에 실제 한 칸의 크기를 곱하여 실제 크기 반영
objp *= square_size

# 카메라 캘리브레이션에 사용할 3D-2D 좌표 쌍 리스트 초기화
objpoints = [] # 실제 세계의 3D 점들 
imgpoints = [] # 이미지 평면의 2D 점들 

# 현재 실행 중인 파일의 디렉토리를 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
image_pattern = os.path.join(base_dir, "images/calibration_images/left*.jpg")
# 여러 장의 체크보드 이미지를 정렬하여 순서대로 불러오기
images = sorted(glob.glob(image_pattern))

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    # 이미지를 BGR 포맷으로 읽어옴
    img = cv2.imread(fname)
    # 코너 검출은 보통 흑백 이미지에서 진행하므로 Grayscale로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 입력 이미지의 크기 저장 (단 한 번만 수행)
    if img_size is None:
        img_size = gray.shape[::-1]

    # 이미지에서 체크보드 코너 찾기 (성공 여부 ret, 코너 좌표 corners 반환)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너가 모두 성공적으로 обнару되면 좌표 저장
    if ret == True:
        # 코너 검출에 성공한 경우 해당 이미지에 대응하는 3D 좌표 추가
        objpoints.append(objp)
        
        # 코너 좌표의 정확도를 서브픽셀 수준으로 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 찾은 코너들을 원본 이미지 위에 그려서 확인용으로 시각화
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Checking Corners', img)
        # 0.1초 동안 대기
        cv2.waitKey(100)

# 모든 창 닫기
cv2.destroyAllWindows()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 얻어진 3D-2D 대응 점들을 바탕으로 카메라 내부 행렬(K)과 왜곡 계수(dist) 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
# [fx, 0, cx / 0, fy, cy / 0, 0, 1] 형태의 내부 파라미터 출력
print(K) 

print("\nDistortion Coefficients:")
# [k1, k2, p1, p2, k3] 형태의 왜곡 계수 출력
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
# 첫 번째 이미지를 샘플로 불러옴
sample_img = cv2.imread(images[0])

# 계산된 카메라 행렬과 왜곡 계수를 사용하여 원본 이미지의 왜곡 보정 수행
dst = cv2.undistort(sample_img, K, dist, None, K)

# 왜곡 보정 전/후 결과 시각화
cv2.imshow('Original Image', sample_img)
cv2.imshow('Undistorted Image', dst)

# 사용자가 키를 입력할 때까지 무한 대기
cv2.waitKey(0)
# 모든 리소스 해제 및 창 닫기
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

# 현재 실행 중인 파일의 절대 경로를 바탕으로 폴더 지정
base_dir = os.path.dirname(os.path.abspath(__file__))
# 변환을 적용할 타겟 이미지의 최종 경로 결합
image_path = os.path.join(base_dir, "images/transformation_images/soccer.jpg")

# -----------------------------
# 1. 이미지 로드
# -----------------------------
# 영상 파일을 BGR 형태로 읽기
img = cv2.imread(image_path)

# 이미지가 제대로 읽히지 않았을 경우의 예외 처리
if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()

# 원본 이미지의 세로(h), 가로(w) 크기를 추출
h, w = img.shape[:2]
# 이미지의 정중앙 좌표를 회전 및 스케일링의 기준점으로 설정
center = (w // 2, h // 2)

# -----------------------------
# 2. 회전 및 크기 조절 (Rotation & Scaling)
# -----------------------------
# 중심(center) 기준으로 30도 반시계방향 회전
angle = 30
# 원본 크기의 0.8배로 축소
scale = 0.8
# 회전과 크기 조절을 동시에 수행하는 2x3 아핀 변환 행렬 도출
M = cv2.getRotationMatrix2D(center, angle, scale)

# -----------------------------
# 3. 평행 이동 (Translation)
# -----------------------------
# x축 방향으로 +80픽셀(우측) 이동
tx = 80
# y축 방향으로 -40픽셀(상단) 이동
ty = -40

# 기존 회전/비율 변환 행렬 M의 3번째 열(평행 이동 성분)에 이동 값을 더해줌
# 이를 통해 회전, 크기 조절, 평행 이동을 하나의 행렬(M)로 통합 완료
M[0, 2] += tx
M[1, 2] += ty

# -----------------------------
# 4. 아핀 변환 적용 (Affine Transformation)
# -----------------------------
# 완성된 변환 행렬 M을 원본 이미지에 최종적으로 매핑(적용)
# 출력 이미지의 크기는 원본과 동일하게 (w, h)로 유지
dst = cv2.warpAffine(img, M, (w, h))

# -----------------------------
# 5. 결과 시각화
# -----------------------------
# 원본 이미지 화면 출력
cv2.imshow('Original Image', img)
# 변환이 적용된 결과물 이미지 출력
cv2.imshow('Transformed Image', dst)

# 창이 바로 닫히지 않고 키보드 입력이 있을 때까지 대기
cv2.waitKey(0)
# 떠있는 OpenCV UI 창들 모두 소멸
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

# 출력 결과 이미지를 저장하기 위한 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. 좌/우 이미지 불러오기
# -----------------------------
# 현재 실행 중인 파일의 상위 폴더 절대 경로 설정
base_dir = Path(__file__).parent.resolve()

# 스테레오 비전 처리에 사용할 좌측, 우측 이미지 로드
left_color = cv2.imread(str(base_dir / "images/left.png"))
right_color = cv2.imread(str(base_dir / "images/right.png"))

# 파일이 제대로 불러와졌는지 예외 처리
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# 스테레오 카메라의 내부 스펙 (초점 거리와 베이스라인 파라미터)
f = 700.0   # 초점 거리 (Focal length)
B = 0.12    # 카메라 간 기준선 거리 (Baseline)

# 영상 내 특정 물체가 위치한 관심 영역(ROI) 지정 (x, y, w, h 형태)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 픽셀 매칭 및 시차 계산은 주로 밝기 정보만 필요하므로 Grayscale 모드로 변환
gray_left = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 2. Disparity(시차) 계산
# -----------------------------
# 스테레오 블록 매칭 알고리즘 객체 생성
# numDisparities: 검색할 최대 시차 수(항상 16의 배수여야 함)
# blockSize: 매칭에 사용할 픽셀 블록의 크기(항상 홀수여야 함)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# 좌/우 영상을 입력하여 초기 시차값을 정수로 도출 (빠른 연산을 위해 16가중치 부여된 값 반환)
disparity_int = stereo.compute(gray_left, gray_right)
# 실제 시차값(픽셀 편차)으로 되돌리기 위해 실수형으로 변환 후 16.0으로 나눔
disparity = disparity_int.astype(np.float32) / 16.0

# -----------------------------
# 3. Depth(깊이) 계산 
# -----------------------------
# 카메라에서 물체까지의 거리를 계산 (수식: Z = fB / d)
# 시차(disparity)가 0보다 큰 유효한 픽셀들에 대해서만 처리 마스크 생성
valid_mask = disparity > 0
# 원본과 동일한 크기의 빈 심도 맵(depth map) 배열 생성
depth_map = np.zeros_like(disparity)
# 유효한 화소 부분에만 삼각 측량 공식(Z = fB / d) 적용
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 4. ROI별 평균 disparity / depth 분석
# -----------------------------
results = {}
for name, (x, y, w, h) in rois.items():
    # ROI 영역에 해당하는 시차와 심도 잘라내기
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    # 위에서 추출한 유효 마스크 값도 동일하게 잘라줌
    roi_valid = valid_mask[y:y+h, x:x+w]
    
    # ROI 영역 내에 유효한 시차 값이 하나라도 있다면 평균 연산 수행
    if np.any(roi_valid):
        avg_disp = np.mean(roi_disp[roi_valid])
        avg_depth = np.mean(roi_depth[roi_valid])
    else:
        # 유효 범위가 없으면 0으로 예외 처리
        avg_disp = 0
        avg_depth = 0
    
    # 분석된 각 물체의 결과를 딕셔너리로 저장
    results[name] = {"avg_disp": avg_disp, "avg_depth": avg_depth}

# -----------------------------
# 5. 결과 콘솔 데이터 출력
# -----------------------------
print(f"{'ROI':<10} | {'Avg Disparity':<15} | {'Avg Depth':<10}")
print("-" * 40)
for name, data in results.items():
    # 소수점 4자리까지만 포맷팅하여 출력
    print(f"{name:<10} | {data['avg_disp']:<15.4f} | {data['avg_depth']:<10.4f}")

# Depth 값이 가장 작은 녀석이 렌즈에서 가장 가까운 객체 (물리적인 거리가 짧음)
closest_roi = min(results.items(), key=lambda item: item[1]['avg_depth'])
# Depth 값이 가장 큰 녀석이 렌즈에서 가장 멀리 있는 객체 (물리적인 거리가 김)
farthest_roi = max(results.items(), key=lambda item: item[1]['avg_depth'])

print(f"\n가장 가까운 객체: {closest_roi[0]} (Avg Depth: {closest_roi[1]['avg_depth']:.4f})")
print(f"가장 먼 객체: {farthest_roi[0]} (Avg Depth: {farthest_roi[1]['avg_depth']:.4f})")

# -----------------------------
# 6. Disparity 시각화 컬러 변경
# -----------------------------
# (가까울수록 편차가 큼 = 편차 큰 것을 붉게 표현)
disp_tmp = disparity.copy()
# 0 이하의 무효한 값들은 비정상 처리 방지용 NaN으로 변환
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

# 상/하위 5% 극단치 아웃라이어 제거를 통한 안정적 정규화
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

# min~max 구간을 0.0~1.0 사이 값으로 Scaling
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
# 범위를 벗어나는 초과/미만 값들을 0과 1로 잘라냄
disp_scaled = np.clip(disp_scaled, 0, 1)

# 컬러맵 적용을 위해 빈 도화지(uint8 0~255 포맷) 생성
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
# 0.0~1.0 값을 255를 곱해 색상 픽셀 값에 매치
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# 컬러맵(JET 타입) 입혀서 파랑~빨강 그라데이션 만들기 (큰 편차가 빨간색)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Depth 시각화 컬러 변경
# -----------------------------
# (가까울수록 거리 Z값이 작음. 작은 값을 붉게 표현을 위해 반전 필요)
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    # 상/하위 5% 극단치를 제외하여 범위 재설정
    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    # min~max 구간을 0.0~1.0 스케일링
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # ***핵심***: Depth는 수치가 클수록 먼 렌즈이고, 멀수록 파랑색이 되게끔 반전
    # 1.0(가장 멂)이 => 0.0(파랑), 0.0(가장 가까움)이 => 1.0(빨강)으로 역전환
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

# 컬러맵(JET) 입히기
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 8. 원본 Left / Right 이미지에 초록색 ROI 박스 표시 확인
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

# 미리 정의된 사각형 좌표들을 순회
for name, (x, y, w, h) in rois.items():
    # Left 이미지 대상 초록색 사각형(ROI)과 라벨(Text) 도형 추가
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Right 이미지도 똑같이 수행
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 9. 컴퓨터내 하드디스크 경로에 결과 이미지 png로 저장
# -----------------------------
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)

# -----------------------------
# 10. OpenCV 창들로 눈에 보이게 출력
# -----------------------------
cv2.imshow("Original Left (with ROIs)", left_vis)
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Depth Map", depth_color)

# 키보드 입력이 이뤄질때까지 대기
cv2.waitKey(0)
# 메모리 상에서 열려있는 Window UI 리소스 전부 해제
cv2.destroyAllWindows()
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
<img width="305" height="112" alt="스크린샷 2026-03-12 오후 4 39 56" src="https://github.com/user-attachments/assets/0289c42e-684c-4213-b9a3-b903c7fecab5" />


