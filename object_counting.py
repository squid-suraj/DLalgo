import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture('bottles.mp4')

unique_ids = set()

while True:
    ret, frame = cap.read()
    # Break the loop if the video has ended
    if not ret:
        break

    results = model.track(frame, classes=[39], persist=True)  # Class 39 is for bottles
    annotated_frame = results[0].plot()

    # Check if any bottles with IDs were detected
    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        for oid in ids:
            unique_ids.add(oid)
    
    # Always display the text and the frame
    cv2.putText(annotated_frame, f"Unique Bottles: {len(unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # FIXED: The typo 'inshow' is now 'imshow'
    # MOVED: This now runs on every frame for a continuous video feed
    cv2.imshow("Annotated Video", annotated_frame)

    # MOVED: The exit check also runs on every frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()