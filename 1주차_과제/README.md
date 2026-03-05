# 📷 OpenCV 1주차 과제 정리

본 저장소는 컴퓨터비전 OpenCV 1주차 과제(1~3)를 수행한 결과를 담고 있습니다.

---

## 📌 과제 1: 원본 이미지와 흑백(Grayscale) 이미지 동시 출력
`01_grayscale_view.py`
원본 컬러 이미지를 불러온 뒤, 이를 흑백 이미지로 변환하고 두 이미지를 나란히 합쳐서 하나의 창에 출력하는 과제입니다.

### 📝 전체 코드
```python
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
```

### 🔑 주요 코드 및 설명
```python
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
```
* **`cv.cvtColor`**: 이미지의 색상 공간을 변환하는 강력한 함수입니다. `COLOR_BGR2GRAY`를 통해 3채널의 컬러 이미지를 1채널 흑백 이미지로 변환합니다.
* **`cv.COLOR_GRAY2BGR`**: 흑백(1채널) 사진을 원본(3채널) 사진과 나란히 이어붙이기 위해서는 두 이미지의 채널 차원 형태가 동일해야 합니다. 따라서 1채널 흑백 사진을 다시 껍데기만 3채널 형태로 늘려주는 역할을 합니다.
* **`np.hstack`**: Numpy 배열로 이루어진 두 이미지를 가로축(수평) 방향으로 이어 붙여 하나의 거대한 배열(이미지)로 만듭니다.

### 🖥 실행 결과 화면
![과제1 결과](./과제1_결과.png)

---

## 📌 과제 2: 마우스 이벤트로 그림판 만들기
`02_painter_brush.py`
OpenCV가 제공하는 마우스 이벤트를 활용하여, 마우스를 클릭하고 드래그하면 화면에 원(붓)이 그려지며 `+`, `-` 키보드로 붓의 크기를 조절할 수 있는 간단한 그림판을 만드는 과제입니다.

### 📝 전체 코드
```python
import cv2 as cv
import sys
import numpy as np

# ======== 초기 설정 ========
#이미지 불러오기
img = cv.imread('girl_laughing.jpg')    

brush_size = 5 # 도장의 기본 크기 (반지름=5픽셀)
# 색상 지정: OPENCV는 BGR순
L_color = (255, 0, 0) # 파란색 (Blue 최대치)
R_color = (0, 0, 255) # 빨간색 (Red 최대치)

# ======== 마우스 동작을 처리해주는 함수 ========
def draw(event, x, y, flags, param):
    # 전역변수 선언: 내가 밖에서 정한 붓 크기(brush_size) 정보를 안에서도 사용할 수 있게 가져옴
    global brush_size
    
    # 1) 좌클릭을 했을 때, OR (마우스 이동 중인데 + 동시에 왼쪽 버튼이 눌려있는 상태일 때)
    if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON):
        # 파란색(L_color)으로 원 그리기
        # cv.circle(그릴화면, (x, y)좌표, 원의반지름크기, 색상, -1은 속을 꽉 채운다는 뜻)
        cv.circle(img, (x, y), brush_size, L_color, -1)
        
    # 2) 우클릭을 했을 때, OR (마우스 이동 중인데 + 동시에 오른쪽 버튼이 눌려있는 상태일 때)
    elif event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON):
        # 빨간색(R_color)으로 꽉 채운 원 그리기
        cv.circle(img, (x, y), brush_size, R_color, -1)
    
    # 그리는 중간중간 화면을 업데이트해서 보여줌
    cv.imshow('Painting', img)

# ======== 윈도우 생성 및 설정 ========
# 'Painting'이라는 이름의 그림판 창을 미리 만듭니다.
cv.namedWindow('Painting')
# 이 창에서 어떤 '마우스 반응'이 있을 때마다 위에서 만든 `draw` 함수를 실행하라고 연결시켜줍니다!
cv.setMouseCallback('Painting', draw)

# ======== 반복문 (메인 루프) ======== 
while True:
    cv.imshow('Painting', img) # 만든 그림판 화면을 띄움
    
    # 키보드 입력을 1밀리초(0.001초) 단위로 감지합니다. 이 입력을 받기 전까진 화면만 띄워 놓음.
    key = cv.waitKey(1) & 0xFF 
    
    # 1) 'q' 버튼을 누르면 그리기 프로그램 반복 종료!
    if key == ord('q'): 
        break
        
    # 2) '+' 버튼, 혹은 '=' 버튼을 누르면 (쉬프트 없이 누른 값 대비용)
    elif key == ord('+') or key == ord('='): 
        # 붓 크기가 커집니다. 단, 너무 커지면 이상하니까 최대 15까지 커지도록 제한(min 사용)
        brush_size = min(15, brush_size + 1) 
        
    # 3) '-' 버튼, 혹은 '_' 버튼을 누르면
    elif key == ord('-') or key == ord('_'): 
        # 붓 크기가 작아집니다. 최소한 크기 1은 되게 제한(max 사용)
        brush_size = max(1, brush_size - 1)  

# 반복문(그림 그리기 작업)이 끝나면 창을 모두 닫아버림.
cv.destroyAllWindows()
```

### 🔑 주요 코드 및 설명
```python
# ======== 윈도우 생성 및 설정 ========
# 'Painting'이라는 이름의 그림판 창을 미리 만듭니다.
cv.namedWindow('Painting')
# 이 창에서 어떤 '마우스 반응'이 있을 때마다 위에서 만든 `draw` 함수를 실행하라고 연결시켜줍니다!
cv.setMouseCallback('Painting', draw)

... (함수 내부) ...
    # 1) 좌클릭을 했을 때, OR (마우스 이동 중인데 + 동시에 왼쪽 버튼이 눌려있는 상태일 때)
    if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON):
        # 파란색(L_color)으로 원 그리기
        # cv.circle(그릴화면, (x, y)좌표, 원의반지름크기, 색상, -1은 속을 꽉 채운다는 뜻)
        cv.circle(img, (x, y), brush_size, L_color, -1)
```
* **`cv.setMouseCallback`**: 특정 창('Painting') 안에서 마우스 움직임이나 클릭이 감지될 때마다, 내가 지정한 함수(`draw`)를 자동으로 호출하도록 연결시켜주는 역할을 합니다.
* **`event`와 `flags`를 이용한 드래그 감지**: 마우스가 그냥 이동하는 것(`EVENT_MOUSEMOVE`)과 알트/컨트롤/마우스버튼 등의 상태 정보가 담긴 `flags`(`EVENT_FLAG_LBUTTON`: 좌클릭이 유지되고 있음)를 조합하여 '드래그(Drag)' 동작을 구현했습니다.
* **`cv.circle`**: 화면의 `(x, y)` 좌표에 반지름이 `brush_size`인 원을 그립니다. 두께 자리에 들어간 `-1`은 원의 내부를 빈틈없이 꽉 채워 그리겠다는 뜻입니다.

### 🖥 실행 결과 화면
![과제2 그림판 결과](./과제2_결과.png)

---

## 📌 과제 3: 마우스 드래그를 이용한 관심영역(ROI) 추출 및 저장
`03_roi_selector.py`
사진 위에서 원하는 영역을 마우스로 드래그해 빨간색 테두리로 표시하고, 해당 부분만을 잘라내(Cropping) 별도의 창으로 띄운 뒤 파일로 저장까지 할 수 있는 기능입니다.

### 📝 전체 코드
```python
import cv2 as cv
import sys

# 1. 이미지 로드
# 'image.jpg' 파일을 읽어옵니다. 파일이 코드와 같은 폴더에 있어야 합니다.
img = cv.imread('girl_laughing.jpg') 

if img is None:
    sys.exit('파일을 찾을 수 없습니다. 경로를 확인하세요.')

# 원본 이미지를 따로 보관합니다. 
# 사각형이 그려진 img를 초기화하거나, 깨끗한 영역을 저장할 때 사용합니다.
original_img = img.copy()

# 전역 변수 초기화
ix, iy = -1, -1      # 마우스 클릭 시작 좌표
drawing = False      # 마우스 클릭 상태 확인
roi = None           # 선택된 영역(Region of Interest) 저장 변수

# 마우스 이벤트 콜백 함수 정의
def select_roi(event, x, y, flags, param):
    global ix, iy, drawing, img, roi
    
    # 마우스 왼쪽 버튼을 눌렀을 때 (시작점)
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y # 클릭한 지점의 좌표 저장
        
    # 마우스 왼쪽 버튼을 뗐을 때 (끝점 및 영역 확정)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        
        # 원본 이미지 위에 선택 영역을 나타내는 빨간색 사각형을 그립니다.
        # (시작좌표), (끝좌표), (색상 BGR), (두께)
        cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)
        
        # 실제 이미지 슬라이싱을 통해 영역을 잘라냅니다.
        # numpy 슬라이싱 규칙: [y시작:y끝, x시작:x끝]
        # min/max를 사용하면 역방향으로 드래그해도 정상적으로 잘립니다.
        roi = original_img[min(iy, y):max(iy, y), min(ix, x):max(ix, x)]
        
        # 잘라낸 영역이 유효하다면 별도의 창에 띄워줍니다.
        if roi.size > 0:
            cv.imshow('Cropped ROI', roi)

# 윈도우 생성 및 마우스 콜백 함수 연결
cv.namedWindow('Select ROI')
cv.setMouseCallback('Select ROI', select_roi)

print("--- 사용 방법 ---")
print("1. 마우스 드래그: 영역 선택")
print("2. 'r' 키: 선택 영역 초기화 (다시 그리기)")
print("3. 's' 키: 선택한 영역을 파일로 저장")
print("4. 'q' 키: 프로그램 종료")

# 키보드 입력을 기다리는 무한 루프
while True:
    cv.imshow('Select ROI', img) # 현재 이미지 상태 표시
    
    key = cv.waitKey(1) & 0xFF # 1ms 동안 키 입력 대기
    
    # 'r' 키를 누르면 (Reset)
    if key == ord('r'):
        img = original_img.copy()     # 사각형이 그려지지 않은 원본으로 복구
        roi = None                    # 저장된 영역 데이터 초기화
        if cv.getWindowProperty('Cropped ROI', 0) >= 0: # 창이 열려있다면
            cv.destroyWindow('Cropped ROI')            # 결과 창 닫기
        print("이미지가 초기화되었습니다.")

    # 's' 키를 누르면 (Save)
    elif key == ord('s'):
        if roi is not None and roi.size > 0:
            # 선택된 영역(roi)을 현재 폴더에 이미지 파일로 저장합니다.
            cv.imwrite('captured_roi.jpg', roi)
            print("선택 영역이 'captured_roi.jpg'로 저장되었습니다.")
        else:
            print("저장할 영역이 없습니다. 먼저 마우스로 드래그하세요.")

    # 'q' 키를 누르면 (Quit)
    elif key == ord('q'):
        break

# 모든 창 닫기
cv.destroyAllWindows()
```

### 🔑 주요 코드 및 설명
```python
        # 원본 이미지 위에 선택 영역을 나타내는 빨간색 사각형을 그립니다.
        # (시작좌표), (끝좌표), (색상 BGR), (두께)
        cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)
        
        # 실제 이미지 슬라이싱을 통해 영역을 잘라냅니다.
        # numpy 슬라이싱 규칙: [y시작:y끝, x시작:x끝]
        # min/max를 사용하면 역방향으로 드래그해도 정상적으로 잘립니다.
        roi = original_img[min(iy, y):max(iy, y), min(ix, x):max(ix, x)]
...
        if roi is not None and roi.size > 0:
            # 선택된 영역(roi)을 현재 폴더에 이미지 파일로 저장합니다.
            cv.imwrite('captured_roi.jpg', roi)
```
* **`cv.rectangle`**: 시작점 `(ix, iy)`부터 현재 마우스를 뗀 점 `(x, y)`까지 이어지는 직사각형을 그립니다. `(0, 0, 255)`는 BGR 기준 빨간색을 의미합니다.
* **`Numpy 이미지 슬라이싱`**: OpenCV의 이미지는 Numpy의 다차원 행렬(Matrix) 구조를 그대로 사용합니다. 따라서 `배열[y축시작:y축끝, x축시작:x축끝]` 형태로 원하는 부분을 뚝 떼어낼 수 있습니다. `min/max`를 사용해 사용자가 오른쪽 위에서 왼쪽 아래 등 역방향으로 드래그를 하더라도 인덱스 에러가 발생하지 않도록 강제 교정하여 잘라냅니다.
* **`cv.imwrite`**: 슬라이싱 되어 `roi` 변수에 담긴 이미지 데이타 배열을 실제 `.jpg` 같은 그림 파일 형태로 컴퓨터(하드디스크)에 저장시키는 함수입니다.

### 🖥 실행 결과 화면
![과제3 ROI 결과 1](./과제3_결과1.png)
![과제3 ROI 결과 2](./과제3_결과2.png)
![과제3 ROI 결과 3](./과제3_결과3.png)
