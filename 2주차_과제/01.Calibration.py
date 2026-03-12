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

    # 이미지에서 체크보드 코너 찾기 (힌트 반영: cv2.findChessboardCorners 적용)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너가 발견되면 좌표 저장 (힌트 반영: 코너 검출에 실패한 이미지는 자동 제외됨)
    if ret == True:
        objpoints.append(objp)
        
        # 코너 좌표 정밀화 (Sub-pixel accuracy)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # (선택 사항) 코너 그리기 시각화
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Checking Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# cv2.calibrateCamera()를 사용하여 카메라 행렬 K와 왜곡 계수 dist를 구함 (요구사항 반영)
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K) # [fx, 0, cx / 0, fy, cy / 0, 0, 1] 형태의 내부 파라미터 

print("\nDistortion Coefficients:")
print(dist) # [k1, k2, p1, p2, k3] 형태의 왜곡 계수 

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
# 첫 번째 이미지를 대상으로 왜곡 보정 결과 확인 
sample_img = cv2.imread(images[0])
# cv2.undistort()를 사용하여 왜곡 보정 적용 (요구사항 반영)
dst = cv2.undistort(sample_img, K, dist, None, K)

# 결과 비교 시각화
cv2.imshow('Original Image', sample_img)
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()