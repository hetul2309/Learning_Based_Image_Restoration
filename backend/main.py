import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference import ModelManager
from backend.metrics import evaluate_metrics, calculate_psnr, calculate_ssim

app = FastAPI(
    title="Image Restoration API",
    description="Backend API for Deep Learning Image Restoration & Quality Evaluation",
    version="1.0.0"
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def base64_to_pil(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

@app.get("/api/models")
def get_models():
    """Returns list of 4 available PyTorch restoration models."""
    return {"models": model_manager.get_available_models()}

@app.post("/api/restore")
async def restore_image(
    model_id: str = Form("cnn"),
    optimizer: str = Form("adam"),
    image_b64: Optional[str] = Form(None),
    degraded_b64: Optional[str] = Form(None),
    purpose: Optional[str] = Form("general")
):
    """
    Directly restores the uploaded image using selected PyTorch model architecture.
    No artificial degradation added.
    """
    try:
        b64_str = image_b64 or degraded_b64
        if not b64_str:
            raise HTTPException(status_code=400, detail="No image provided")

        input_pil = base64_to_pil(b64_str)
        restored_pil = model_manager.restore_image(model_id=model_id, degraded_pil=input_pil)

        return {
            "model_id": model_id,
            "optimizer": optimizer,
            "purpose": purpose,
            "restored_b64": pil_to_base64(restored_pil)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount built React static files from frontend/dist
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
react_dist = os.path.join(project_root, "frontend", "dist")
if not os.path.exists(react_dist):
    react_dist = os.path.join(project_root, "frontend_react", "dist")

if os.path.exists(react_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(react_dist, "assets")), name="react_assets")

@app.get("/")
def read_root():
    if os.path.exists(react_dist):
        return FileResponse(os.path.join(react_dist, "index.html"))
    return {"message": "Image Restoration API backend running."}
