import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-v3kp.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Device Selection
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF = DEVICE == "cuda"

print(f"Using Device: {DEVICE}")

# -----------------------------
# Load Model
# -----------------------------
model = YOLO("yolo11n.pt")      # Faster than yolo11s
model.to(DEVICE)

# JPEG Quality
JPEG_QUALITY = 70

# Resize Resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client Connected")

    try:
        while True:

            # Receive image bytes
            data = await websocket.receive_bytes()

            # Decode JPEG
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Resize frame
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Inference
            with torch.inference_mode():

                results = model(
                    frame,
                    imgsz=640,
                    device=DEVICE,
                    half=HALF,
                    verbose=False,
                    conf=0.35
                )

            # Draw detections
            annotated = results[0].plot()

            # Encode back to JPEG
            success, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            if not success:
                continue

            # Send annotated image
            await websocket.send_bytes(encoded.tobytes())

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("Error:", e)

    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
