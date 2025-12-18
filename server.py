# server.py (simple version)
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once
model = YOLO("yolo11s.pt")  # make sure this file is in same folder


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Receives JPEG frames, runs YOLO, sends back annotated JPEG frames."""
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # Receive raw JPEG bytes
            data = await websocket.receive_bytes()

            # Convert bytes → NumPy array → BGR frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text("ERROR: Frame decode failed")
                continue

            # YOLO detection
            results = model(frame)
            annotated = results[0].plot()  # draw boxes

            # Encode annotated frame back to JPEG
            success, encoded = cv2.imencode(".jpg", annotated)

            if not success:
                await websocket.send_text("ERROR: Encode failed")
                continue

            # Send annotated frame
            await websocket.send_bytes(encoded.tobytes())

    except Exception as e:
        print("Client disconnected:", e)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
