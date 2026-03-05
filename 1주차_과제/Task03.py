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