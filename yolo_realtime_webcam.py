import cv2
from ultralytics import YOLO

# Print OpenCV version
print(f"OpenCV version: {cv2.__version__}")

# Load the latest YOLO11 model (choose model size: n, s, m, l, x)
model = YOLO("yolo11s.pt")  # Alternatives: 'yolo11n.pt', 'yolo11m.pt', etc.

# Open the laptop webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit the detection window.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run detection; YOLO expects BGR frames (default for OpenCV)
        results = model.predict(source=frame, show=False, stream=False)

        # Overlay boxes; results[0].plot() returns BGR image
        annotated_frame = results[0].plot()

        # Display result
        cv2.imshow("YOLO11 Real-time Detection", annotated_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")
