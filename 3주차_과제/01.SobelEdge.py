import cv2 as cv # OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import numpy as np # 수치 연산을 위한 numpy 라이브러리를 np라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt # 그래프나 이미지를 시각화하기 위해 matplotlib의 pyplot을 불러옵니다.
import os # 운영체제의 파일 및 디렉토리 관리 기능을 사용하기 위해 os 라이브러리를 불러옵니다.

# 현재 실행 중인 파일의 절대 경로를 바탕으로 폴더 경로를 지정합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))
# 사용할 이미지(edgeDetectionImage.jpg)의 최종 절대 경로를 결합하여 문자열로 생성합니다.
image_path = os.path.join(base_dir, "images/edgeDetectionImage.jpg")

# 지정된 경로의 이미지 파일을 BGR 형식의 픽셀 배열로 읽어옵니다.
img = cv.imread(image_path)
# 만약 이미지를 정상적으로 불러오지 못해 img 변수가 None이라면,
if img is None:
    # 에러 메시지를 출력합니다.
    print(f"Error: Could not read image at {image_path}")
    # 프로그램을 강제 종료합니다.
    exit()

# Matplotlib으로 띄우기 위해 BGR 색상 배열을 RGB 색상 공간으로 변환합니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 1. 이미지를 그레이스케일로 변환
# 그레이스케일(흑백) 형태로 변경하여 연산 속도를 높이고 노이즈 처리를 수월하게 합니다.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Sobel 필터를 사용하여 x축과 y축 방향의 에지를 검출
# 가로(x축) 방향 미분을 수행하여 수직선 에지를 찾습니다 (CV_64F로 데이터 손실 방지, 커널사이즈 3)
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
# 세로(y축) 방향 미분을 수행하여 수평선 에지를 찾습니다 (커널사이즈 3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 3. 에지 강도 계산
# x축 방향 및 y축 방향의 기울기 벡터를 결합해 최종적인 에지 강도를 계산합니다.
magnitude = cv.magnitude(sobel_x, sobel_y)

# 4. 에지 강도 이미지를 시각화하기 위해 uint8로 변환
# 64비트 실수형을 절대값을 취하고 0~255 사이 8비트 정수형(uint8)으로 변환해 화면 표시에 적합하게 만듭니다.
magnitude = cv.convertScaleAbs(magnitude)

# 5. 결과 시각화 (원본 이미지와 에지 강도 이미지 나란히 출력)
# 가로 12인치, 세로 6인치 크기로 새로운 도화지(Figure) 객체를 생성합니다.
plt.figure(figsize=(12, 6))

# 1행 2열의 구조 중 1번째 공간에 그래프를 설정합니다.
plt.subplot(1, 2, 1)
# 원본 색상 변환 이미지(RGB)를 화면에 그립니다.
plt.imshow(img_rgb)
# 해당 서브플롯의 제목을 'Original Image'로 설정합니다.
plt.title('Original Image')
# 그림 주변의 축(x, y 눈금자)을 숨깁니다.
plt.axis('off')

# 1행 2열의 구조 중 2번째 공간에 그래프를 설정합니다.
plt.subplot(1, 2, 2)
# 에지 강도를 구한 영상 값을 흑백 맵(gray)으로 표시합니다.
plt.imshow(magnitude, cmap='gray')
# 해당 서브플롯의 제목을 'Edge Magnitude'로 설정합니다.
plt.title('Edge Magnitude')
# 그림 주변의 축을 숨깁니다.
plt.axis('off')

# 그래프의 여백을 촘촘하게(레이아웃 침범 방지) 자동 조정합니다.
plt.tight_layout()
# 완성된 Matplotlib 이미지를 컴퓨터에 '과제1_결과.png'라는 이름으로 잘리지 않게 저장합니다.
plt.savefig('과제1_결과.png', dpi=300, bbox_inches='tight')
# 열려있는 플롯(Plot) 창과 관련된 메모리를 안전하게 소멸시킵니다.
plt.close()
