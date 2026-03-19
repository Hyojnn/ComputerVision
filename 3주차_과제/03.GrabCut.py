import cv2 as cv # 컴퓨터 비전 처리를 위한 필수 라이브러리 OpenCV를 임포트합니다.
import numpy as np # 빠르고 강력한 다차원 배열 연산을 돕는 numpy를 불러옵니다.
import matplotlib.pyplot as plt # 차트나 사진 시각화의 정석인 matplotlib 기능들을 가져옵니다.
import os # 파일 및 폴더 탐색을 위한 운영 체제 라이브러리 os를 로드합니다.

# 이 코드를 돌리고 있는 파이썬 파일 기준 절대경로를 구하여 폴더명만 담습니다.
base_dir = os.path.dirname(os.path.abspath(__file__))
# 3번째 과제 사진인 'coffee cup.JPG'의 완벽한 맥 환경 파일 경로를 디렉토리명과 합쳐서 반환합니다.
image_path = os.path.join(base_dir, "images/coffee cup.JPG")

# 지정된 경로의 이미지 파일을 BGR 형식의 픽셀 배열로 읽어옵니다.
img = cv.imread(image_path)
# 만약 이미지를 정상적으로 불러오지 못해 img 변수가 None이라면,
if img is None:
    # 에러 메시지를 출력합니다.
    print(f"Error: Could not read image at {image_path}")
    # 프로그램을 강제 종료합니다.
    exit()

# 화면상에 왜곡된 푸릇푸릇한 색상을 방지하기 위해 일반적 포맷인 RGB로 일치화시킵니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 1. 초기 마스크 및 배경 모델, 전경 모델 생성
# 원본 이미지의 가로세로 해상도(크기)에 부합하는 빈 공간인 0(uint8) 배열의 블랙 마스크를 창조해냅니다.
mask = np.zeros(img.shape[:2], np.uint8)
# GMM 방식 알고리즘이 내부적으로 학습할 배경(Bgd) 파라미터 보관용 빈 실수 배열 행렬을 만듭니다.
bgdModel = np.zeros((1, 65), np.float64)
# GMM 방식의 객체 혹은 전경(Fgd) 상태와 분포 가중치를 저장할 65차원 공간의 파라미터 모델을 0으로 꽉 채웁니다.
fgdModel = np.zeros((1, 65), np.float64)

# 2. 초기 사각형 영역은 (x, y, width, height) 형식으로 설정
# height(높이)와 width(폭) 정보를 슬라이싱을 통해 tuple 형태로 나누어 받습니다.
h, w = img.shape[:2]
# 커피 컵 영역을 적절하게 감싸줄 포착 사각형(배경은 쳐내고 객체만 들어올) 크기 값을 설정 및 지정합니다.
rect = (w // 6, h // 6, w * 2 // 3, h * 2 // 3)

# 3. cv.grabCut()를 사용하여 대화식 분할 알고리즘을 수행
# 이 사각형(rect)을 초기 조건으로 5번의 반복 연산(iterCount)동안 통계 학습 모델이 작동하며 배경을 분할합니다.
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

# 4. 마스크를 사용하여 이미지 복원 및 원본 이미지에서 배경 제거
# GC_BGD(0), GC_PR_BGD(2) 등 지워질 뒷배경(0) 요소를 제외한 전경 부분만 1의 값인 uint8 이진 형태로 재설정합니다.
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 이렇게 완성된 이진 흑백 마스크 평면 공간을 3차원(`newaxis`)으로 늘려 곱셈하면, 원본 이미지에서 0인 뒷배경은 검은색(삭제)으로 변합니다.
img_extracted = img * mask2[:, :, np.newaxis]
# 결과를 matplotlib을 통해 표시할 수 있게끔 BGR을 RGB로 색 조합 형식을 바꿔줍니다.
img_extracted_rgb = cv.cvtColor(img_extracted, cv.COLOR_BGR2RGB)

# 5. 결과 시각화
# 여러 장을 옆으로 길게 뽑아내기 위해 15x6 비율의 넓직한 판을 깝니다.
plt.figure(figsize=(15, 6))

# 3개의 구획 중 제일 첫 번째 자리를 준비합니다.
plt.subplot(1, 3, 1)
# 필터링 단계를 안 거친 순수 첫 원본 이미지 자체를 묘사합니다.
plt.imshow(img_rgb)
# 사용자가 이 그림이 뭐였는지 알게 상단 제목으로 달아줍니다.
plt.title('Original Image')
# 좌표축 따위의 선들이 분위기를 해치지 않게 가립니다.
plt.axis('off')

# 정중앙의 2번째 서브플롯 영역으로 이동합니다.
plt.subplot(1, 3, 2)
# 전경이 하얗게(1), 원치 않는 배경부분이 새까맣게(0) 걸러진 이진 분할 결과 마스크 맵을 도식화합니다.
plt.imshow(mask2, cmap='gray')
# 마찬가지로 이것이 'Mask Image' 란 영문 타이틀을 작성해 부착합니다.
plt.title('Mask Image')
# 불필요한 x축과 y축 숫자를 안 보이게 끕니다.
plt.axis('off')

# 가장 오른쪽 끄트머리에 세 번째 플롯을 그립니다.
plt.subplot(1, 3, 3)
# 배경이 전부 지워져서 커피 컵 객체 피사체만 남은 RGB 분할 결과 사진을 나타냅니다.
plt.imshow(img_extracted_rgb)
# 사진 위에다 'Extracted Object'라고 이름표를 적어줍니다.
plt.title('Extracted Object')
# 옆의 그림들처럼 경계면 축 텍스트 표시와 라인을 지워버립니다.
plt.axis('off')

# 화면 요소들을 딱딱 포개어 정렬시키고 공백을 자동으로 적절하게 나눠줍니다.
plt.tight_layout()
# 현재 시점까지 그려낸 플롯 도표 객체를 물리적인 파일시스템에 이미지 확장자인 png 포맷으로 저장/생성시킵니다.
plt.savefig('과제3_결과.png', dpi=300, bbox_inches='tight')
# 해당 렌더링된 객체를 프로그램 메모리 상 클리어하게 해제하고 지워 안정화를 도모합니다.
plt.close()
