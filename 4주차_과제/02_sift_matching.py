import cv2 as cv
import matplotlib.pyplot as plt

# 1. 두 이미지 로드
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color83.jpg') # 80 대신 존재하는 83 사용

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
