import cv2 # 영상 관련 처리를 위한 OpenCV 라이브러리 임포트
import mediapipe as mp # 얼굴 랜드마크 추출 등에 사용하는 Mediapipe 라이브러리 임포트
import os # 운영체제 경로 및 디렉토리 관련 처리를 위한 os 모듈 임포트

# 현재 스크립트의 디렉토리를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 실행 중인 파일의 절대 디렉토리 경로
results_dir = os.path.join(current_dir, 'results') # 실행 결과 이미지를 저장할 results 폴더 경로 조합
os.makedirs(results_dir, exist_ok=True) # results 폴더가 없다면 생성 (존재해도 에러 발생 방지)

# 1. Mediapipe의 FaceMesh 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh # FaceMesh 모델 모듈 가져오기
face_mesh = mp_face_mesh.FaceMesh( # FaceMesh 객체 인스턴스 생성
    max_num_faces=1, # 영상에서 찾을 최대 얼굴 개수는 1개로 제한
    refine_landmarks=True, # 눈과 입술 주변 등 추가 세부 랜드마크 추출 옵션 활성화
    min_detection_confidence=0.5, # 얼굴로 인식하는 최소 신뢰도 임계값 설정
    min_tracking_confidence=0.5 # 얼굴 추적 임계값 설정 (신뢰도에 미달하면 다시 찾음)
)

# 2. 웹캠 열기
cap = cv2.VideoCapture(0) # 컴퓨터에 연결된 기본 웹캠(0번 인덱스) 연결 포착

if not cap.isOpened(): # 카메라가 성공적으로 열리지 않았다면
    print("웹캠을 찾을 수 없습니다. 비디오 캡처를 사용할 수 없습니다.") # 오류 메시지 출력

saved_result = False # 처음 한 장의 결과 이미지를 기록했는지 여부를 표시하는 플래그
print("얼굴 랜드마크 추출 시작... (중간에 끄려면 'q' 또는 ESC 키를 누르세요)") # 실행 안내 문구 출력

# 3. 실시간 영상 캡처 루프
while cap.isOpened(): # 카메라에서 정상적으로 입력을 받고 있는 동안 반복
    ret, frame = cap.read() # 카메라로부터 1 프레임씩 캡처
    if not ret: # 프레임을 받아오지 못했다면
        print("연결된 카메라에서 영상을 읽을 수 없습니다.") # 에러 메시지 출력
        break # 무한 반복 루프에서 빠져나옴
        
    # 성능을 위해 이미지를 읽기 전용으로 설정
    frame.flags.writeable = False # 원본 프레임을 임시로 읽기 전용으로 설정 (처리 성능 향상 목적)
    
    # BGR을 RGB로 변환 (Mediapipe는 RGB 이미지를 요구함)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV 표준인 BGR 형식에서 RGB 형식으로 변환
    
    # 프로세싱: 얼굴 랜드마크 검출
    results = face_mesh.process(rgb_frame) # RGB 이미지에서 얼굴과 랜드마크(특징점) 좌표 분석
    
    # 다시 쓰기 가능으로 설정하고 BGR로 변환 상태 유지
    frame.flags.writeable = True # 이미지 배열을 다시 그리기 및 쓰기가 가능하도록 속성 변경
    
    # 4. 검출된 랜드마크 시각화 (frame에 점으로 표시)
    if results.multi_face_landmarks: # 모델이 얼굴과 랜드마크를 1개 이상 찾았다면
        for face_landmarks in results.multi_face_landmarks: # 발견한 얼굴 랜드마크 목록을 순회
            for landmark in face_landmarks.landmark: # 각각의 랜드마크(특징점) 좌표(x, y, z)에 대하여 반복
                # 랜드마크 좌표는 정규화(0~1)되어 있으므로 창 크기에 맞게 스케일 변환
                ih, iw, _ = frame.shape # 현재 이미지의 세로, 가로, 채널 정보를 얻어옴
                x = int(landmark.x * iw) # 0~1사이의 정규화된 x 값을 실제 픽셀 좌표(가로 길이 곱)로 변환
                y = int(landmark.y * ih) # 0~1사이의 정규화된 y 값을 실제 픽셀 좌표(세로 길이 곱)로 변환
                
                # 얼굴 랜드마크를 녹색 점으로 그리기
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1) # 해당 좌표에 점을 표시 (지름 1, 색상:녹색, 내부채움)
                
        # 최초 발견 시 1회 결과 이미지 저장
        if not saved_result: # 아직 결과 이미지를 저장하지 않았다면
            cv2.imwrite(os.path.join(results_dir, '과제2_결과.png'), frame) # 현재 그려진 프레임을 파일로 저장
            saved_result = True # 결과 이미지를 한 번 저장했으므로 True로 변경
            
    # 영상 시각화 터미널 창 출력
    cv2.imshow('Mediapipe FaceMesh', frame) # 결과가 그려진 BGR 이미지를 새 화면 창으로 띄우기
    
    # 5. ESC 키(27) 또는 'q' 키를 누르면 루프 종료
    key = cv2.waitKey(5) & 0xFF # 5ms 동안 키 입력 대기하며 키 코드 받음 (64비트 호환 0xFF 비트연산)
    if key == 27 or key == ord('q'): # 받은 키가 ESC 나 'q' 이면
        print("사용자에 의해 영상이 중간에 종료되었습니다.") # 종료 문구 출력
        break # 비디오 캡처 루프 탈출

# 자원 해제
cap.release() # 프로그램 종료 전, 사용해주던 웹캠 장치를 해제
cv2.destroyAllWindows() # 열어두었던 OpenCV 관련 모든 창을 종료
