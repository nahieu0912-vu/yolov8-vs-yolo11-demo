import cv2
from ultralytics import YOLO

# Load your YOLO model (e.g., the fast Nano version) 
model = YOLO("best.pt")

# Open a video file (or use '0' to display from your webcam)
cap = cv2.VideoCapture("test2/mp4/test.mp4")  # Replace with your video path or use 0 for webcam

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run inference on the current frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame on your screen
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the display loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
