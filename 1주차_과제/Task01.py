import cv2 as cv
import numpy as np
import sys

# 1. 이미지 로드 (컴퓨터에 저장된 이미지를 읽어옴)
# cv.imread() 함수는 이미지 파일을 읽어서 numpy 배열(행렬) 형태로 변환합니다.
img = cv.imread('soccer.jpg') # 'soccer.jpg' 파일을 불러옵니다.

# 이미지가 정상적으로 불러와지지 않았다면(예: 파일 이름 오타, 경로 문제 등)
# img 변수는 None이 되므로 프로그램이 에러 없이 종료되도록 처리합니다.
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 2. 이미지를 그레이스케일(흑백)로 변환
# cv.cvtColor() 함수는 색상 공간을 변환할 때 사용합니다. 
# cv.COLOR_BGR2GRAY 옵션은 컬러(BGR) 이미지를 흑백(GRAY) 사진으로 바꿔줍니다.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 3. 흑백 이미지를 다시 3채널(BGR) 형태로 변환
# 흑백 이미지(gray)는 1채널(명암만 가짐)이고, 컬러 이미지(img)는 3채널(BGR)입니다.
# 두 이미지를 나란히 붙이려면(np.hstack) 채널 수가 똑같아야 하므로, 
# 흑백 사진의 형태를 억지로 3채널로 늘려줍니다. (색깔이 생기는 건 아니고 그릇 모양만 맞추는 작업)
gray_3channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 4. 원본 사진과 흑백 사진을 가로로(수평으로) 이어붙이기
# np.hstack()은 튜플()로 묶인 배열들을 가로축 기준으로 나란히 연결해주는 역할을 합니다.
res = np.hstack((img, gray_3channel))

# 5. 결과 화면 출력 
# 'Original and Grayscale'이라는 창을 띄워서 합쳐놓은 이미지(res)를 보여줍니다.
cv.imshow('Original and Grayscale', res)

# 사용자가 아무 키보드 키를 누를 때까지 기다립니다. (0을 넣으면 무한정 대기)
cv.waitKey(0) 

# 사용자가 키를 눌러서 루프를 빠져나오면, 열려있는 모든 OpenCV 창을 닫고 프로그램을 종료합니다.
cv.destroyAllWindows()