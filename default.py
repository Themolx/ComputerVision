from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.predict(
    source=0,
    show=True,
    conf=0.5,        # Confidence threshold 0-1
    iou=0.7,         # NMS IOU threshold
    show_labels=True,  # Show labels
    show_conf=True,   # Show confidences
    boxes=True,       # Show boxes
    line_width=2      # Box thickness
)
