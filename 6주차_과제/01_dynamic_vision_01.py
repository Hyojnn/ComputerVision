import cv2
import numpy as np
import os
from sort import Sort

# 현재 스크립트 경로를 기준으로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# YOLOv3 설정 파일과 가중치 파일 경로
cfg_path = os.path.join(current_dir, 'yolov3.cfg')
weights_path = os.path.join(current_dir, 'yolov3.weights')

# SORT 추적기 초기화
tracker = Sort()

# YOLO 모델 로드
print("YOLOv3 모델 로딩 중...")
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = net.getLayerNames()

try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    # 예전 버전 OpenCV 호환용
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 비디오 파일 오픈
video_path = os.path.join(current_dir, 'slow_traffic_small.mp4')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

saved_result = False
print("비디오 처리 시작...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    height, width, channels = frame.shape
    
    # 1. YOLO를 이용한 객체 검출
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # 신뢰도가 일정 수준 이상인 경우에만 검출 인정
            if confidence > 0.5:
                # 차량(2)이나 사람(0) 등 여러 객체를 원활히 검출하기 위해 임계치만 사용
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # 경계 상자 좌표 계산
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # NMS(Non-Maximum Suppression)를 통해 겹치는 박스 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # 2. 검출 결과를 SORT 추적기 입력 형태로 변환 [x1, y1, x2, y2, score]
    detections = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            detections.append([x, y, x + w, y + h, confidences[i]])
            
    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    
    # 3. SORT 객체 추적 업데이트
    tracks = tracker.update(detections)
    
    # 4. 결과 시각화
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        
        # 상태에 따른 고유 ID 표시 (비디오에 바운딩 박스 그리기)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID: {track_id}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    cv2.imshow("SORT Multi-Object Tracking", frame)
    
    # 첫 프레임 결과를 README 용으로 저장
    if not saved_result and len(tracks) > 0:
        cv2.imwrite(os.path.join(results_dir, '과제1_결과.png'), frame)
        saved_result = True
        
    # ESC 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == 27: 
        break
        
cap.release()
cv2.destroyAllWindows()
