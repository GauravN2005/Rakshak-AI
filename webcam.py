from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, stream=True)

    # Draw results
    for r in results:
        annotated_frame = r.plot()

    # Show output
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
