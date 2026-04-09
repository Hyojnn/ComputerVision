import cv2
import numpy as np
import sys
import os

# add L06 to path to import sort
sys.path.append(os.path.join(os.path.dirname(__file__), 'L06'))
from sort import Sort

def main():
    # Load YOLOv3 network
    net = cv2.dnn.readNet("L06/yolov3.weights", "L06/yolov3.cfg")
    layer_names = net.getLayerNames()
    
    # Check OpenCV version to use appropriate getUnconnectedOutLayers() return format
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
    # Load coco.names
    with open("L06/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    # Open video
    cap = cv2.VideoCapture("L06/slow_traffic_small.mp4")
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get video properties for writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize SORT tracker
    tracker = Sort()

    # Define the codec and create VideoWriter object
    os.makedirs('results', exist_ok=True)
    out = cv2.VideoWriter('results/01_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out_det in outs:
            for detection in out_det:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # apply Non-Max Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Prepare info for SORT
        # Format for SORT: [x1, y1, x2, y2, score]
        dets = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                dets.append([x, y, x + w, y + h, confidences[i]])
        
        dets = np.array(dets)
        if len(dets) == 0:
            dets = np.empty((0, 5))

        # Update tracker
        trackers = tracker.update(dets)

        # Draw tracking results
        for d in trackers:
            x1, y1, x2, y2, track_id = [int(v) for v in d]
            
            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            text = f"ID: {track_id}"
            cv2.putText(frame, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        
        # print progress
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames")
        frame_idx += 1

    cap.release()
    out.release()
    print("Tracking completed. Output saved to results/01_result.mp4")

if __name__ == "__main__":
    main()
