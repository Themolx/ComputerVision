from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from ultralytics import YOLO
import json
import base64
import asyncio
import time
from collections import deque
import logging
import os
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="YOLO Vision Interactive")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class ModelManager:
    def __init__(self):
        self.models = {
            'yolo11n': {'path': 'yolo11n.pt', 'type': 'detect', 'loaded': None},
            'yolo11n-seg': {'path': 'yolo11n-seg.pt', 'type': 'segment', 'loaded': None},
            'yolo11n-pose': {'path': 'yolo11n-pose.pt', 'type': 'pose', 'loaded': None}
        }
        self.current_model = 'yolo11n'
        self.load_model(self.current_model)

    def load_model(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        if self.models[model_name]['loaded'] is None:
            logger.info(f"Loading model: {model_name}")
            try:
                self.models[model_name]['loaded'] = YOLO(self.models[model_name]['path'])
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise
        
        self.current_model = model_name
        return self.models[model_name]['loaded']

    def get_current_model(self):
        return self.models[self.current_model]['loaded']

    def get_model_type(self):
        return self.models[self.current_model]['type']

class VideoProcessor:
    def __init__(self):
        self.active = False
        self.settings = {
            "confidence": 0.25,
            "iou": 0.45,
            "resolution": "720p",
            "show_labels": True
        }
        self.capture = None
        self.fps_buffer = deque(maxlen=30)
        self.model_manager = ModelManager()
        
    def initialize_capture(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise RuntimeError("Could not start camera.")
            self.set_resolution(self.settings["resolution"])
    
    def set_resolution(self, resolution):
        if resolution == "720p":
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        elif resolution == "1080p":
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    def release_capture(self):
        if self.capture:
            self.capture.release()
            self.capture = None
    
    def update_settings(self, settings):
        self.settings.update(settings)
        if "resolution" in settings and self.capture:
            self.set_resolution(settings["resolution"])

    async def process_frame(self, frame):
        start_time = time.time()
        
        model = self.model_manager.get_current_model()
        model_type = self.model_manager.get_model_type()
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=self.settings["confidence"],
            iou=self.settings["iou"],
            verbose=False
        )
        
        # Process detections
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
        
        # Calculate processing statistics
        process_time = (time.time() - start_time) * 1000
        self.fps_buffer.append(process_time)
        memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create annotated frame
        annotated_frame = results[0].plot()
        
        # Add performance overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            annotated_frame, 
            f"{model_type.upper()}: {process_time:.1f}ms", 
            (10, 30), font, 1, (255, 255, 255), 2
        )
        
        return detections, annotated_frame, process_time, memory

# Initialize video processor
video_processor = VideoProcessor()

# Routes
@app.get("/")
async def read_index():
    try:
        with open('static/index.html', 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index file not found")

@app.get("/api/models")
async def get_models():
    """Get list of available models"""
    return list(video_processor.model_manager.models.keys())

@app.post("/api/settings")
async def update_settings(settings: dict):
    """Update video processor settings"""
    try:
        video_processor.update_settings(settings)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/switch_model")
async def switch_model(data: dict):
    """Switch the current detection model"""
    try:
        video_processor.model_manager.load_model(data["model"])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    logger.info("New WebSocket connection attempting to connect...")
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        # Initialize video capture
        video_processor.initialize_capture()
        video_processor.active = True
        
        while True:
            if video_processor.active:
                # Capture frame
                ret, frame = video_processor.capture.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                try:
                    # Process frame
                    detections, annotated_frame, process_time, memory = await video_processor.process_frame(frame)
                    
                    # Encode frame to base64
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send data to client
                    await websocket.send_json({
                        "frame": frame_base64,
                        "detections": detections,
                        "process_time": process_time,
                        "memory": memory,
                        "timestamp": time.time()
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    continue
                
            # Add small delay to control frame rate
            await asyncio.sleep(0.01)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")
        video_processor.release_capture()

# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    video_processor.release_capture()
    logger.info("Server shutting down, released video capture")

if __name__ == "__main__":
    print("🚀 Starting YOLO Vision Interactive Installation...")
    print("📍 Open http://localhost:8000 in your browser")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
