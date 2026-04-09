import cv2 # OpenCV 라이브러리 임포트 (영상 처리 및 컴퓨터 비전 기능 제공)
import numpy as np # 수치 연산을 위한 NumPy 라이브러리 임포트
import os # 운영체제와 상호작용하기 위한 os 모듈 임포트 (파일 경로 처리 등)
from sort import Sort # 객체 추적을 위한 SORT 알고리즘 클래스 임포트

# 현재 스크립트 경로를 기준으로 설정
current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 실행 중인 스크립트 파일의 절대 경로가 속한 디렉토리 반환
results_dir = os.path.join(current_dir, 'results') # 결과 이미지를 저장할 'results' 폴더 경로 생성
os.makedirs(results_dir, exist_ok=True) # 결과 폴더가 없다면 생성 (exist_ok=True로 이미 존재하면 무시)

# YOLOv3 설정 파일과 가중치 파일 경로
cfg_path = os.path.join(current_dir, 'yolov3.cfg') # YOLOv3 모델의 네트워크 구조 설정 파일 경로
weights_path = os.path.join(current_dir, 'yolov3.weights') # YOLOv3 모델의 가중치 파일 경로

# SORT 추적기 초기화
tracker = Sort() # 객체 추적기 SORT 인스턴스 생성

# YOLO 모델 로드
print("YOLOv3 모델 로딩 중...") # 모델 로딩 시작을 알리는 메시지 출력
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path) # Darknet 형식의 YOLOv3 모델과 가중치 불러오기
layer_names = net.getLayerNames() # 네트워크의 모든 레이어 이름 가져오기

try: # 최신 OpenCV 버전의 방식 시도
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] # 출력 레이어(Unconnected 레이어)의 이름 리스트 생성
except: # 에러 발생 시 예전 버전 OpenCV 호환 방식(이전 버전은 리스트 안의 리스트 형태로 반환함)으로 대체
    # 예전 버전 OpenCV 호환용
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # 이전 버전 OpenCV를 고려하여 출력 레이어 이름 추출

# 비디오 파일 오픈
video_path = os.path.join(current_dir, 'slow_traffic_small.mp4') # 입력 비디오 파일 경로 설정
cap = cv2.VideoCapture(video_path) # OpenCV 비디오 캡처 객체 생성

if not cap.isOpened(): # 비디오 파일이 정상적으로 열렸는지 확인
    print("비디오 파일을 열 수 없습니다.") # 열 수 없다면 오류 메시지 출력
    exit() # 프로그램 종료

saved_result = False # 첫 번째 프레임 결과 저장을 확인하기 위한 플래그 변수 초기화
print("비디오 처리 시작... (중간에 끄려면 'q' 또는 ESC 키를 누르세요)") # 처리 시작 안내 메시지

while True: # 비디오의 매 프레임별 반복 처리
    ret, frame = cap.read() # 비디오로부터 한 프레임 읽기 (ret는 성공 여부, frame은 이미지 데이터)
    if not ret: # 프레임을 제대로 읽지 못했거나 영상이 끝났으면
        break # 반복문(영상 처리 루프) 탈출
        
    height, width, channels = frame.shape # 현재 프레임 이미지의 세로, 가로, 채널 수 정보 가져오기
    
    # 1. YOLO를 이용한 객체 검출
    # 이미지를 YOLO 네트워크 입력 형태(blob)로 변환: 스케일링(1/255), 크기 조정(416x416), BGR->RGB 변경 안함
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob) # 변환된 blob을 네트워크 입력으로 설정
    outs = net.forward(output_layers) # 네트워크 순전파를 통해 예측 결과(출력 레이어의 결과) 가져오기
    
    class_ids = [] # 검출된 객체의 클래스 ID들을 저장할 리스트
    confidences = [] # 검출된 객체들의 신뢰도(확률)를 저장할 리스트
    boxes = [] # 검출된 객체들의 경계 상자(바운딩 박스) 좌표를 저장할 리스트
    
    for out in outs: # 각 출력 레이어의 결과 순회
        for detection in out: # 각 검출된 박스 정보 순회
            scores = detection[5:] # 처음 5개 값 제외한 나머지가 각 클래스에 속할 확률(점수) 배열
            class_id = np.argmax(scores) # 가장 확률이 높은 클래스의 인덱스 추출
            confidence = scores[class_id] # 객체에 대한 최대 신뢰도 값 추출
            
            # 신뢰도가 일정 수준 이상인 경우에만 검출 인정
            if confidence > 0.5: # 신뢰도가 0.5보다 큰 객체만 의미있는 검출로 인정
                # 차량(2)이나 사람(0) 등 여러 객체를 원활히 검출하기 위해 임계치만 사용
                center_x = int(detection[0] * width) # 검출된 객체의 중심 x 좌표 계산 (원점 이미지 크기에 비례)
                center_y = int(detection[1] * height) # 검출된 객체의 중심 y 좌표 계산
                w = int(detection[2] * width) # 객체의 실제 너비 계산
                h = int(detection[3] * height) # 객체의 실제 높이 계산
                
                # 경계 상자 좌표 계산
                x = int(center_x - w / 2) # 경계 상자의 좌상단 x 좌표 계산
                y = int(center_y - h / 2) # 경계 상자의 좌상단 y 좌표 계산
                
                boxes.append([x, y, w, h]) # 계산된 경계 상자 [x, y, w, h] 정보를 boxes에 추가
                confidences.append(float(confidence)) # 해당 객체의 신뢰도 값을 추가 (NMS 사용 시 float 형 요구)
                class_ids.append(class_id) # 해당 객체의 클래스 ID를 추가
                
    # NMS(Non-Maximum Suppression)를 통해 겹치는 박스 제거
    # score 임계값 0.5, NMS 임계값 0.4 설정으로 중복되는 경계 상자 제거 후 최종 인덱스 반환
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # 2. 검출 결과를 SORT 추적기 입력 형태로 변환 [x1, y1, x2, y2, score]
    detections = [] # SORT 추적기에 입력할 형태로 좌표를 변환해 임시 저장할 리스트
    if len(indexes) > 0: # NMS를 통과하여 유효한 객체가 1개라도 있는 경우
        for i in indexes.flatten(): # 유효 객체들의 인덱스 순회
            x, y, w, h = boxes[i] # 유효한 박스의 원본 정보 추출
            # SORT 형식에 맞게 좌상단 및 우하단 좌표와 신뢰도로 변환 및 저장
            detections.append([x, y, x + w, y + h, confidences[i]])
            
    # detections 리스트를 NumPy 배열로 변환 (빈 경우 (0,5) 크기의 빈 배열 생성)
    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    
    # 3. SORT 객체 추적 업데이트
    # 프레임의 바운딩 박스들을 SORT 추적기로 전달하여 업데이트된 ID와 위치(tracks) 반환
    tracks = tracker.update(detections)
    
    # 4. 결과 시각화
    for track in tracks: # 추적기에서 반환된 각각의 추적 정보 순회
        x1, y1, x2, y2, track_id = track.astype(int) # 얻은 위치와 ID를 정수로 형변환
        
        # 상태에 따른 고유 ID 표시 (비디오에 바운딩 박스 그리기)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 현재 프레임에 객체 경계 상자(녹색, 두께 2) 그리기
        text = f"ID: {track_id}" # 표시할 텍스트에 ID 값 삽입
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # 박스 위에 ID 텍스트 표시
        
    cv2.imshow("SORT Multi-Object Tracking", frame) # 추적 결과가 그려진 프레임을 화면 이름 지정하여 표시
    
    # 첫 프레임 결과를 README 용으로 저장
    if not saved_result and len(tracks) > 0: # 결과 기록 플래그가 거짓이고, 추적된 객체가 있으면
        cv2.imwrite(os.path.join(results_dir, '과제1_결과.png'), frame) # 현재까지 작업된 프레임을 이미지로 처음 1회 추출/저장
        saved_result = True # 저장 완료 상태로 변경
        
    # ESC 키 또는 'q' 키를 누르면 종료
    key = cv2.waitKey(30) & 0xFF # 30ms 동안 키보드 입력 대기
    if key == 27 or key == ord('q'): # ESC키(키코드 27) 혹은 'q'가 입력되면
        print("사용자에 의해 영상이 중간에 종료되었습니다.") # 터미널에 메시지 출력
        break # 루프 종료
        
cap.release() # 비디오 캡처 객체의 리소스 해제
cv2.destroyAllWindows() # OpenCV 기능으로 생성된 모든 화면 창 닫기
