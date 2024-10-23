from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from ultralytics import YOLO
import json
import base64
import asyncio
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLO model
model = YOLO('yolo11n.pt')

# HTML content for the test page
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>YOLO WebSocket Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #videoFeed { max-width: 800px; width: 100%; border: 1px solid #ccc; }
            #status { margin: 10px 0; padding: 10px; background: #f0f0f0; }
            #detections { margin-top: 10px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>YOLO WebSocket Test</h1>
        <div id="status">Connecting...</div>
        <img id="videoFeed" alt="Video Feed">
        <div id="detections"></div>

        <script>
            const status = document.getElementById('status');
            const videoFeed = document.getElementById('videoFeed');
            const detectionsDiv = document.getElementById('detections');
            
            function connect() {
                const ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = function() {
                    status.textContent = 'Connected';
                    status.style.background = '#d4edda';
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    videoFeed.src = 'data:image/jpeg;base64,' + data.frame;
                    detectionsDiv.textContent = JSON.stringify(data.detections, null, 2);
                };

                ws.onclose = function() {
                    status.textContent = 'Disconnected - Reconnecting...';
                    status.style.background = '#fff3cd';
                    setTimeout(connect, 1000);
                };

                ws.onerror = function(error) {
                    status.textContent = 'Error: ' + error;
                    status.style.background = '#f8d7da';
                };
            }

            connect();
        </script>
    </body>
</html>
"""

class VideoProcessor:
    def __init__(self):
        self.active = False
        self.settings = {
            "confidence": 0.25,
            "iou": 0.45
        }
        self.capture = None
    
    def initialize_capture(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise RuntimeError("Could not start camera.")
    
    def release_capture(self):
        if self.capture:
            self.capture.release()
            self.capture = None
    
    async def process_frame(self, frame):
        results = model.predict(
            source=frame,
            conf=self.settings["confidence"],
            iou=self.settings["iou"],
            verbose=False
        )
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].tolist()
                c = box.cls
                conf = box.conf.item()
                class_name = model.names[int(c)]
                
                height, width = frame.shape[:2]
                x1, y1, x2, y2 = b
                detection = {
                    "x": (x1 / width) * 100,
                    "y": (y1 / height) * 100,
                    "width": ((x2 - x1) / width) * 100,
                    "height": ((y2 - y1) / height) * 100,
                    "class": class_name,
                    "confidence": conf
                }
                detections.append(detection)
        
        return detections, results[0].plot()

# Create video processor instance
video_processor = VideoProcessor()

@app.get("/", response_class=HTMLResponse)
async def get():
    return html

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("New WebSocket connection attempting to connect...")
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        video_processor.initialize_capture()
        video_processor.active = True
        
        while True:
            if video_processor.active:
                ret, frame = video_processor.capture.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                try:
                    detections, annotated_frame = await video_processor.process_frame(frame)
                    
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    await websocket.send_json({
                        "frame": frame_base64,
                        "detections": detections
                    })
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
                
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket connection closed")
        video_processor.release_capture()

if __name__ == "__main__":
    print("Starting YOLO webcam server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
