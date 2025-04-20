from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn
import shutil
from pathlib import Path
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS for mobile browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("static/index.html")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model(image)
    results_json = results.pandas().xyxy[0].to_dict(orient="records")
    return {"results": results_json}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)