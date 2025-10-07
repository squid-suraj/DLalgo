import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture('peoplemoving.mp4')

while True:
    ret, frame = cap.read()
    # ADDED: Check if the video has ended
    if not ret:
        break

    # MOVED: Create a copy of the frame to draw on for every iteration
    annotated_frame = frame.copy()
    
    results = model.track(source=frame, classes=[0], persist=True)

    # The results object is a list, but for a single frame, it usually has one element
    if results and results[0]:
        r = results[0]
        # Check if masks, boxes, and IDs are detected
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            # ADDED: .cpu() for compatibility before converting to numpy
            masks = r.masks.data.cpu().numpy() # FIXED: Typo 'maks' to 'masks'
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()

            for i, mask in enumerate(masks):
                person_id = ids[i]
                x1, y1, x2, y2 = boxes[i].astype(int)
                
                # Resize the mask to the original frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                
                # Find contours from the resized mask
                contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw the segmentation contours and ID
                cv2.drawContours(annotated_frame, contours, -1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'ID: {int(person_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # MOVED: Display the frame outside the inner loops to prevent freezing
    cv2.imshow('Segmented Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()