import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 두 이미지 로드 (샘플 이미지 img1.jpg, img2.jpg 선택)
img1 = cv.imread('images/img2.jpg') # 캔버스의 기준이 될 이미지 (왼쪽 사진)
img2 = cv.imread('images/img3.jpg') # 변환되어 오른쪽에 붙을 이미지 (오른쪽 사진)

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

# 매칭 결과 이미지 생성 (시각화용)
img_matches = cv.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 4. 호모그래피 행렬 계산
# 최소 4개의 매칭점이 필요함
if len(good_matches) > 4:
    # 매칭점의 좌표 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # RANSAC을 이용한 호모그래피 계산
    # img2의 좌표를 변환하여 img1의 좌표계로 맞춤 (이렇게 하면 오른쪽 사진이 양수 좌표로 확장되므로 잘리지 않습니다)
    H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    
    # 5. 한 이미지를 변환하여 정렬 (파노라마)
    # img1을 img2 평면으로 투영. 
    # 결과 이미지 캔버스 크기: 너비는 두 이미지 너비의 합, 높이는 두 이미지 중 최대 높이
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    out_w, out_h = w1 + w2, max(h1, h2)
    
    # img2 변환
    warped_img2 = cv.warpPerspective(img2_rgb, H, (out_w, out_h))
    
    # 변환된 결과 그림 위에 img1을 겹쳐 그리기 (경계선 스무딩 - Alpha Blending 적용)
    panorama = warped_img2.copy()
    
    # img1과 img2가 겹치는 폭(w1)에 대해 x축 기준 선형 그라데이션(알파 값 1.0 -> 0.0) 생성
    alpha_gradient = np.linspace(1, 0, w1).reshape(1, w1, 1)
    
    img1_area = img1_rgb
    img2_area = warped_img2[0:h1, 0:w1]
    
    # img2가 할당된 유효한 픽셀 마스크(검은색 빈 공간 제외)
    valid_mask2 = np.any(img2_area > 0, axis=2, keepdims=True)
    
    # 겹치는 부분 혼합: img1은 갈수록 투명해져 0에 수렴하고 img2가 서서히 드러나게 함
    blended = (img1_area * alpha_gradient + img2_area * (1 - alpha_gradient)).astype(np.uint8)
    
    # img2 데이터가 없던 곳은 그대로 img1 원본 복사, 겹치는 구역은 혼합된 블렌딩 적용
    panorama[0:h1, 0:w1] = np.where(valid_mask2, blended, img1_area)
    
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
    plt.savefig('results/과제3_결과.png')
    print('결과가 results/과제3_결과.png에 저장되었습니다.')
    plt.show()
else:
    print('Not enough matches are found - {}/4'.format(len(good_matches)))
