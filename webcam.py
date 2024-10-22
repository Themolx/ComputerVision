from ultralytics import YOLO
import cv2
import time

# Load the model
model = YOLO('yolo11n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the desired resolution
desired_width = 1280  # Replace with your desired width
desired_height = 720  # Replace with your desired height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Variables for FPS calculation
prev_time = time.time()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 output
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (desired_width, desired_height))

# Color definitions
TEXT_COLOR = (255, 255, 255)  # White color for text
BOX_COLOR = (0, 255, 0)       # Green color for bounding boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction on the frame
    results = model.predict(
        frame,
        conf=0.1,        # Confidence threshold
        iou=0.5,         # NMS IOU threshold
        verbose=False    # Suppress unnecessary output
    )

    # Visualize the results on the frame
    annotated_frame = results[0].plot(
        boxes=True,
        labels=True,
        conf=True,
        line_width=2
    )

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Overlay FPS on the frame
    cv2.putText(
        annotated_frame,
        f'FPS: {fps:.2f}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    # Count the number of detections
    num_detections = len(results[0].boxes)
    cv2.putText(
        annotated_frame,
        f'Detections: {num_detections}',
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    # Add a custom message
    cv2.putText(
        annotated_frame,
        'Press Q to Exit',
        (10, desired_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Show the annotated frame
    cv2.imshow('YOLO Detection', annotated_frame)

    # Exit when 'Q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
