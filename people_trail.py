import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture('peoplemoving.mp4')

id_map = {}
next_id = 0

trail = defaultdict(lambda: deque(maxlen=30))
appear = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    res = model.track(frame, classes=[0], persist=True)
    annotated_frame = res[0].plot()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.cpu().numpy()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            appear[oid] += 1

            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))
                
                # --- Drawing Logic ---
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'ID: {sid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                # ✨ ADDED: Draw the trail
                if len(trail[oid]) > 1:
                    points = np.array(trail[oid], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

    cv2.imshow("tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()