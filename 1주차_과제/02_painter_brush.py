import cv2 as cv
import sys
import numpy as np

# ======== 초기 설정 ========
# 그림판 배경이 될 빈 이미지 캔버스(도화지) 생성
# np.ones: 지정한 크기((세로500, 가로600, 3채널))의 배열을 만들고 모두 1로 채웁니다. 
# 그 위에 255를 곱해주면 모든 값이 (255, 255, 255)가 되어 완전한 흰색 배경 이미지가 완성됩니다.
img = np.ones((500, 600, 3), np.uint8) * 255 

brush_size = 5 # 도장의 기본 크기 (반지름=5픽셀)
# 색상 지정: OpenCV는 이미지 색상을 RGB가 아니라 BGR(Blue, Green, Red) 순서로 다룹니다!
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