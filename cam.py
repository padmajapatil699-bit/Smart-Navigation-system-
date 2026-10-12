from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

# store last detected object
last_object = None

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break
    # run detection
    results = model(frame, verbose=False)

    frame_height, frame_width = frame.shape[:2]

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)

            label = model.names[cls]

            # calculate position
            object_center = (x1 + x2) / 2

            if object_center < frame_width/3:
                position = "LEFT"
            elif object_center < 2*frame_width/3:
                position = "CENTER"
            else:
                position = "RIGHT"

            # estimate distance
            area = (x2-x1)*(y2-y1)

            if area > 60000:
                distance = "VERY CLOSE"
            elif area > 25000:
                distance = "CLOSE"
            else:
                distance = "FAR"

            # print only when object changes
            if label != last_object:

                print("\nObstacle detected")
                print("Object:", label)
                print("Position:", position)
                print("Distance:", distance)

                last_object = label

            # draw box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()