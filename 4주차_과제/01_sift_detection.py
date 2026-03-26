import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 로드
img = cv.imread('images/mot_color70.jpg')
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
plt.savefig('results/과제1_결과.png')
print('결과가 results/과제1_결과.png에 저장되었습니다.')
plt.show()
