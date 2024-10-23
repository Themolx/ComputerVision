from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('yolo11n-seg.pt')

# Set the video source to a YouTube URL
source = 1#'https://youtu.be/M6bm6yRKshA'

# Initialize variables
first_frame = True
out = None

# Run prediction on the video source with streaming
results = model.predict(
    source=source,
    show=False,          # We'll handle showing frames manually
    stream=True,         # Return a generator
    conf=0.2,            # Confidence threshold
    iou=0.5,             # NMS IOU threshold
    verbose=False,        # Suppress unnecessary output
    stream_buffer=False,
)

for result in results:
    # Get the annotated frame
    annotated_frame = result.plot()

    # Initialize VideoWriter on the first frame
    if first_frame:
        height, width = annotated_frame.shape[:2]
        fps = 15  # Set FPS to 30 or adjust as needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        first_frame = False

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Show the annotated frame
    cv2.imshow('YOLO Detection', annotated_frame)

    # Exit when 'Q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
if out:
    out.release()
cv2.destroyAllWindows()

