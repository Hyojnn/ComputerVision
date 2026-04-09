import cv2
import mediapipe as mp
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', action='store_true', help='Run for a few frames and save a snapshot')
    args = parser.parse_args()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    os.makedirs('results', exist_ok=True)
    frame_count = 0
    saved = False

    print("Press ESC to exit...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = face_mesh.process(rgb_frame)

        # Draw the face mesh annotations on the image.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        cv2.imshow('MediaPipe FaceMesh', frame)

        # In test mode, save after a few frames to ensure webcam has adjusted
        if args.test_mode and frame_count == 30:
            cv2.imwrite('results/02_result.jpg', frame)
            print("Snapshot saved to results/02_result.jpg")
            saved = True
            break
        
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # If it wasn't test mode but we exit, let's still save the last frame if we haven't
    if not saved and frame is not None:
        cv2.imwrite('results/02_result.jpg', frame)
        print("Final frame saved to results/02_result.jpg")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
