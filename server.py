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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load YOLO model (CPU SAFE)
# =========================
MODEL_PATH = "yolov8n.pt"   # 🔥 FAST & CPU FRIENDLY

print("🔄 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded")

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

            # Decode frame
            frame = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                print("⚠️ Frame decode failed")
                continue

            # 🔥 YOLO inference (Render optimized)
            results = model(
                frame,
                imgsz=320,      # SMALL SIZE
                conf=0.35,
                device="cpu"
            )

            print("📦 Detections:", len(results[0].boxes))

            annotated_frame = results[0].plot()

            # Encode JPEG
            success, buffer = cv2.imencode(".jpg", annotated_frame)
            if not success:
                print("⚠️ Encode failed")
                continue

            # Send back frame
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("❌ Client disconnected")

    except Exception as e:
        print("🔥 WebSocket error:", e)

# =========================
# Health Check (Render needs this)
# =========================
@app.get("/")
def health():
    return {"status": "Backend running 🚀"}

# =========================
# Run Server
# =========================
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )
