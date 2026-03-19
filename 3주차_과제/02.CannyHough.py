import cv2 as cv # OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import numpy as np # 수치 연산을 위한 numpy 라이브러리를 np라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt # 시각화를 위한 matplotlib의 pyplot 기능을 불러옵니다.
import os # 파일 및 폴더 경로 작업을 위해 os 파이썬 내장 라이브러리를 불러옵니다.

# 현재 디렉토리의 절대 경로를 계산하여 base_dir에 저장합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))
# 불러올 사진 파일(dabo.jpg)의 올바른 통합 경로를 문자열로 만듭니다.
image_path = os.path.join(base_dir, "images/dabo.jpg")

# 설정된 경로의 원본 이미지를 BGR 형태로 디코딩하여 numpy 배열로 읽어 들입니다.
img = cv.imread(image_path)
# 파일 인식이나 경로 설정의 문제로 이미지가 불러와지지 않은 경우
if img is None:
    # 사용자에게 안내 메시지를 콘솔로 출력합니다.
    print(f"Error: Could not read image at {image_path}")
    # 프로그램을 즉시 종료합니다.
    exit()

# Matplotlib의 정상적 출력 처리를 위해 BGR 순서를 RGB 순서로 배열 변환합니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# 추후 붉은색 직선을 그려줄 도화지용 원본 이미지를 메모리에 미리 복사해 둡니다.
img_line = img.copy()

# 1. 이미지를 그레이스케일로 변환
# 3채널의 색상값을 가진 이미지를 1채널 흑백 이미지로 바꿔 연산 부하를 크게 줄입니다.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Canny 에지 검출을 사용하여 에지 맵 생성 
# 최소 임계값(100), 최대 임계값(200)을 넘어가는 얇고 선명한 테두리를 흰색 선으로 뽑아냅니다.
edges = cv.Canny(gray, threshold1=100, threshold2=200)

# 3. 허프 변환을 사용하여 이미지에서 직선 검출
# 확률적 허프 변환(HoughLinesP)으로, 간격 허용도(maxLineGap)와 최소 길이(minLineLength)에 맞는 직선의 좌표들을 얻어옵니다.
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# 4. 검출된 직선을 원본 이미지에서 빨간색으로 표시
# 직선이 단 1개라도 배열 데이터로 무사히 검출되었다면,
if lines is not None:
    # 이중 리스트 형태(lines) 안의 각 직선의 x,y 좌표 배열들을 순회 반복합니다.
    for line in lines:
        # 직선 1개의 (시작 x, 시작 y, 끝 x, 끝 y) 값을 언패킹해 가져옵니다.
        x1, y1, x2, y2 = line[0]
        # 미리 복제해 둔 사진 도화지(img_line)에 두께가 2 픽셀인 (0=B, 0=G, 255=R) 빨간 직선을 긋습니다.
        cv.line(img_line, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 화면에 직선 그려진 걸 표출해주기 위해 이미지의 포맷을 RGB로 한 번 다시 변경합니다.
img_line_rgb = cv.cvtColor(img_line, cv.COLOR_BGR2RGB)

# 5. 결과 시각화
# 좌우로 시원하게 배치하기 위해 가로 12, 세로 6인치 크기 설정된 레이아웃 창을 엽니다.
plt.figure(figsize=(12, 6))

# 1행 2열로 화면을 분할하고 그중 좌측에 해당하는 플롯 공간을 택합니다.
plt.subplot(1, 2, 1)
# 아무 선도 없어 깨끗한 원본 RGB 이미지를 그림판에 그립니다.
plt.imshow(img_rgb)
# 서브플롯 영역 최상단에 'Original Image' 영문 제목을 달아줍니다.
plt.title('Original Image')
# 불필요한 테두리 축 정보들을 제거하여 깔끔하게 만듭니다.
plt.axis('off')

# 우측 영역 두 번째 플롯을 지목합니다.
plt.subplot(1, 2, 2)
# 앞서 빨간 선을 새롭게 그린 이미지 버전을 배열로 밀어 넣습니다.
plt.imshow(img_line_rgb)
# 서브플롯의 제목(title)으로 'Detected Lines'를 지정합니다.
plt.title('Detected Lines')
# 눈금과 축 라벨을 시각화 과정에서 보이지 않도록 없앱니다.
plt.axis('off')

# 화면 구성을 자율적으로 정돈하여 두 이미지의 겹침을 방지합니다.
plt.tight_layout()
# 작성한 플로팅 이미지 전체를 300 해상도(dpi)를 가진 고품질 png 사진 파일로 데스크탑에 다운로드 저장합니다.
plt.savefig('과제2_결과.png', dpi=300, bbox_inches='tight')
# 사용 완료 후 즉시 내부 도화지를 닫고 리소스 자원 메모리를 반환 및 종료합니다.
plt.close()
