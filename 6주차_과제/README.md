# 📷 OpenCV 6주차 과제 정리

본 저장소는 컴퓨터비전 OpenCV 6주차 실습 과제(1~2)를 수행한 결과를 담고 있습니다. 다중 객체 추적과 얼굴 랜드마크 검출을 구현하였습니다.

---

## 📌 과제 1: SORT 알고리즘을 활용한 다중 객체 추적기 구현
`01_dynamic_vision_01.py`
사전 훈련된 YOLOv3 객체 검출 모델을 사용하여 비디오 프레임 내 객체를 검출하고, SORT 알고리즘을 사용해 각 객체를 실시간으로 추적 및 ID를 시각화하는 프로그램입니다.

### 📝 전체 코드
```python
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
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 비디오 파일 오픈
video_path = os.path.join(current_dir, 'slow_traffic_small.mp4')
cap = cv2.VideoCapture(video_path)

saved_result = False
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
            
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # NMS를 통해 겹치는 박스 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # 2. SORT 추적기 입력 형태로 변환 [x1, y1, x2, y2, score]
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID: {track_id}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    cv2.imshow("SORT Multi-Object Tracking", frame)
    
    if not saved_result and len(tracks) > 0:
        cv2.imwrite(os.path.join(results_dir, '과제1_결과.png'), frame)
        saved_result = True
        
    if cv2.waitKey(30) & 0xFF == 27: 
        break
        
cap.release()
cv2.destroyAllWindows()
```

### 🔑 주요 코드 및 설명
```python
tracker = Sort()
...
detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
tracks = tracker.update(detections)
```
* **`Sort()`**: 객체들의 위치를 추적하기 위해 칼만 필터(Kalman Filter)와 헝가리안 매칭 알고리즘을 사용하는 SORT 추적기를 초기화합니다.
* **`tracker.update(detections)`**: 이전 프레임에서 발견된 객체들의 위치 정보와 현재 프레임에서 검출된 객체들을 연관 계산시켜서 동일한 객체는 고유한 `track_id`를 계속 유지하도록 업데이트합니다. YOLOv3만으로는 매 프레임별 객체의 연속성을 알아낼 수 없기에 해당 추적 알고리즘을 사용합니다.

### 🖥 실행 결과 화면
![과제1 결과](./results/과제1_결과.png)

---

## 📌 과제 2: Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화
`02_dynamic_vision_02.py`
웹캠을 통해 캡쳐되는 실시간 영상에서 Google Mediapipe의 `FaceMesh` 모듈을 사용하여 얼굴의 468개 랜드마크를 추출하고 화면에 실시간 렌더링하는 과제입니다.

### 📝 전체 코드
```python
import cv2
import mediapipe as mp
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# 1. Mediapipe FaceMesh 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. 웹캠 열기
cap = cv2.VideoCapture(0)

saved_result = False
# 3. 실시간 영상 캡처 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame.flags.writeable = False
    
    # BGR을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 얼굴 랜드마크 검출 실행
    results = face_mesh.process(rgb_frame)
    
    frame.flags.writeable = True
    
    # 4. 검출된 랜드마크 시각화
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # 정규화 좌표를 스케일 변환
                ih, iw, _ = frame.shape
                x = int(landmark.x * iw)
                y = int(landmark.y * ih)
                
                # 얼굴 랜드마크를 녹색 점으로 그리기
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
        if not saved_result:
            cv2.imwrite(os.path.join(results_dir, '과제2_결과.png'), frame)
            saved_result = True
            
    cv2.imshow('Mediapipe FaceMesh', frame)
    
    # 5. ESC 키 입력 시 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

### 🔑 주요 코드 및 설명
```python
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
...
    results = face_mesh.process(rgb_frame)
...
                x = int(landmark.x * iw)
                y = int(landmark.y * ih)
```
* **`FaceMesh()`**: 468개(혹은 더 상세한)의 주요 얼굴 특징점 렌더링에 최적화된 Mediapipe 객체를 생성합니다.
* **`process(rgb_frame)`**: 입력받은 웹캠 프레임(RGB) 상에서 얼굴의 점들을 검출합니다.
* **비율 변환 (`x, y` 계산)**: Mediapipe가 반환하는 `landmark.x`와 `landmark.y`는 `0.0~1.0`의 정규화된 상댓값이므로, 비디오의 원본 해상도(`iw`, `ih`)를 곱해주어 `cv2.circle()` 함수가 좌표를 인식할 수 있게 변환 해주어야 합니다.

### 🖥 실행 결과 화면
![과제2_결과](./results/과제2_결과.png)
