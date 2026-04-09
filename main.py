"""
Food Safety AI  –  FastAPI Backend
===================================
• Serves the static frontend (static/)
• POST /api/predict  →  returns { label, confidence, tip }
• Uses a mock predictor when no model file exists.
  Drop your trained model at  models/food_model.keras
  and uncomment the TensorFlow block to switch to real inference.
"""

import asyncio
import os
import random
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "food_model.keras"
IMG_SIZE   = (224, 224)

# Health tips pool
FRESH_TIPS = [
    "Great choice! Fresh food is rich in vitamins and nutrients. Enjoy your meal!",
    "This food looks fresh and nutritious. Eating fresh supports a healthy immune system.",
    "Fresh food detected! A balanced diet with fresh produce reduces disease risk.",
    "Looks delicious and safe! Fresh foods are packed with antioxidants and fibre.",
]
ROTTEN_TIPS = [
    "⚠️ Warning: Consuming spoiled food can lead to food poisoning and severe gastric illness.",
    "⚠️ Danger: This food shows signs of spoilage. Discard it immediately to avoid health risks.",
    "⚠️ Caution: Spoiled food contains harmful bacteria like Salmonella and E. coli. Do not eat.",
    "⚠️ Alert: Food contamination detected. Eating this could cause nausea, vomiting, or worse.",
]

# ─────────────────────────────────────────────────────────────
#  Model Loading  (TensorFlow / Keras)
# ─────────────────────────────────────────────────────────────
model = None

def load_model():
    """Try to load a saved Keras model. Returns None if unavailable."""
    if not MODEL_PATH.exists():
        return None
    try:
        import tensorflow as tf          # noqa: F401  (optional dependency)
        m = tf.keras.models.load_model(str(MODEL_PATH))
        print(f"[INFO] Model loaded from {MODEL_PATH}")
        return m
    except Exception as exc:
        print(f"[WARN] Could not load model: {exc}. Falling back to mock predictor.")
        return None


# ─────────────────────────────────────────────────────────────
#  Prediction Logic
# ─────────────────────────────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> "np.ndarray":
    """Decode image bytes → normalised (1, 224, 224, 3) numpy array."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
    return np.expand_dims(arr, axis=0)               # add batch dim


async def predict_freshness(file_bytes: bytes, filename: str = "") -> dict:
    """
    Core prediction function.

    Returns
    -------
    dict with keys:
        label       – "EDIBLE" | "NOT EDIBLE"
        confidence  – float in [0, 1]  (confidence for the winning class)
        tip         – health tip string
    """
    if model is not None:
        # ── Real inference ──────────────────────────────────────
        tensor = preprocess_image(file_bytes)
        # Assumes model outputs [[fresh_prob, rotten_prob]]
        preds  = model.predict(tensor, verbose=0)[0]
        fresh_prob  = float(preds[0])
        rotten_prob = float(preds[1])
    else:
        # ── Mock inference (2-second simulated delay) ───────────
        await asyncio.sleep(2)
        
        lower_name = filename.lower()
        if "fresh" in lower_name:
            rotten_prob = random.uniform(0.01, 0.49)
        elif "rotten" in lower_name or "spoiled" in lower_name:
            rotten_prob = random.uniform(0.51, 0.99)
        else:
            rotten_prob = random.random()          # uniform in [0, 1]
            
        fresh_prob  = 1.0 - rotten_prob

    # ── Classification logic ────────────────────────────────────
    if rotten_prob > 0.5:
        label      = "NOT EDIBLE"
        confidence = rotten_prob
        tip        = random.choice(ROTTEN_TIPS)
    else:
        label      = "EDIBLE"
        confidence = fresh_prob
        tip        = random.choice(FRESH_TIPS)

    return {"label": label, "confidence": round(confidence, 4), "tip": tip}


# ─────────────────────────────────────────────────────────────
#  FastAPI Application
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Food Safety AI",
    description="Classifies food images as Edible or Not Edible using MobileNetV2/ResNet50.",
    version="1.0.0",
)

# Mount static files (CSS, JS, images)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()
    mode  = "Real model" if model else "Mock predictor (no model file found)"
    print(f"[INFO] Food Safety AI started - mode: {mode}")


@app.get("/", response_class=FileResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an image upload and return freshness classification."""
    # Validate MIME type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    # Size guard (10 MB)
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 10 MB.")

    try:
        result = await predict_freshness(file_bytes, getattr(file, "filename", ""))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return JSONResponse(content=result)


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "mode": "real" if model else "mock",
    }
