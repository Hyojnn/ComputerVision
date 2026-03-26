import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 두 이미지 로드 (샘플 이미지 img1.jpg, img2.jpg 선택)
img1 = cv.imread('images/img1.jpg')
img2 = cv.imread('images/img2.jpg')

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
    # img1의 좌표를 변환하여 img2의 좌표계로 맞춤
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    # 5. 한 이미지를 변환하여 정렬 (파노라마)
    # img1을 img2 평면으로 투영. 
    # 결과 이미지 캔버스 크기: 너비는 두 이미지 너비의 합, 높이는 두 이미지 중 최대 높이
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    out_w, out_h = w1 + w2, max(h1, h2)
    
    # img1 변환
    warped_img1 = cv.warpPerspective(img1_rgb, H, (out_w, out_h))
    
    # 변환된 결과 그림 위에 img2를 겹쳐 그리기
    # img2는 변환 없이 원점(0,0)에 배치되는 기준 이미지라고 가정했을 경우.
    # 하지만 img1에서 img2로 변환하는 H를 구했으므로 img2가 기준이 됩니다.
    panorama = warped_img1.copy()
    
    # img2를 왼쪽에 덮어씌움
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
    plt.savefig('results/과제3_결과.png')
    # plt.show()
else:
    print('Not enough matches are found - {}/4'.format(len(good_matches)))
