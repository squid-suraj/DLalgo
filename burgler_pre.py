import cv2

cap = cv2.VideoCapture(0)

frames = []
gap = 5
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

    if len(frames) > gap + 1:
        frames.pop(0)

    cv2.putText(frame, f"Frame Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    count += 1

    if len(frames) > gap:
        diff = cv2.absdiff(frames[0], frames[-1])
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        motion = any(cv2.contourArea(c) > 1000 for c in contours)

        if motion:
            cv2.putText(frame, "Motion Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(f"motion_frame_{count}.jpg", frame)
            print(f"Saved: motion_frame_{count}.jpg")

            cv2.imshow("Motion Detection", frame)
            count += 1

            if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
                break

cap.release()
cv2.destroyAllWindows()

