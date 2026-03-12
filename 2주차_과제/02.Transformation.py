import cv2
import numpy as np
import os

# 현재 실행 중인 파일의 디렉토리를 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "images/transformation_images/soccer.jpg")

# 1. 이미지 로드
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()

# 원본 이미지 크기
h, w = img.shape[:2]
center = (w // 2, h // 2)

# -----------------------------
# 2. 회전 및 크기 조절 (Rotation & Scaling)
# -----------------------------
# 중심(center) 기준으로 30도 회전, 크기는 0.8로 조절
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

# 키 입력 대기 후 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
