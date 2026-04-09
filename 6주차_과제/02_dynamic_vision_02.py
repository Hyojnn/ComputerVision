import cv2
import mediapipe as mp
import os

# 현재 스크립트의 디렉토리를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# 1. Mediapipe의 FaceMesh 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 찾을 수 없습니다. 비디오 캡처를 사용할 수 없습니다.")

saved_result = False
print("얼굴 랜드마크 추출 시작... (종료하려면 ESC를 누르세요)")

# 3. 실시간 영상 캡처 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("연결된 카메라에서 영상을 읽을 수 없습니다.")
        break
        
    # 성능을 위해 이미지를 읽기 전용으로 설정
    frame.flags.writeable = False
    
    # BGR을 RGB로 변환 (Mediapipe는 RGB 이미지를 요구함)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 프로세싱: 얼굴 랜드마크 검출
    results = face_mesh.process(rgb_frame)
    
    # 다시 쓰기 가능으로 설정하고 BGR로 변환 상태 유지
    frame.flags.writeable = True
    
    # 4. 검출된 랜드마크 시각화 (frame에 점으로 표시)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # 랜드마크 좌표는 정규화(0~1)되어 있으므로 창 크기에 맞게 스케일 변환
                ih, iw, _ = frame.shape
                x = int(landmark.x * iw)
                y = int(landmark.y * ih)
                
                # 얼굴 랜드마크를 녹색 점으로 그리기
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
        # 최초 발견 시 1회 결과 이미지 저장
        if not saved_result:
            cv2.imwrite(os.path.join(results_dir, '과제2_결과.png'), frame)
            saved_result = True
            
    # 영상 시각화 터미널 창 출력
    cv2.imshow('Mediapipe FaceMesh', frame)
    
    # 5. ESC 키를 누르면 루프 종료 (27은 ESC ASCII 번호)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
