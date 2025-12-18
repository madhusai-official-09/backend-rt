# server.py
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI()

# =========================
# CORS (Render + Vercel)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Render & Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load YOLO model ONCE
# =========================
MODEL_PATH = "yolo11s.pt"

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully")

# =========================
# WebSocket Endpoint
# =========================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    try:
        while True:
            # Receive JPEG bytes
            data = await websocket.receive_bytes()

            # Decode JPEG → frame
            frame = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                await websocket.send_text("ERROR: Frame decode failed")
                continue

            # Run YOLO
            results = model(frame, conf=0.4)
            annotated_frame = results[0].plot()

            # Encode back to JPEG
            success, buffer = cv2.imencode(".jpg", annotated_frame)
            if not success:
                await websocket.send_text("ERROR: JPEG encode failed")
                continue

            # Send annotated frame
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("❌ Client disconnected")

    except Exception as e:
        print("🔥 WebSocket error:", e)

# =========================
# Health Check (IMPORTANT for Render)
# =========================
@app.get("/")
def health():
    return {"status": "Backend running 🚀"}

# =========================
# Local run
# =========================
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )
