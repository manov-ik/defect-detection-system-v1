from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from PIL import Image
import io
import base64

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv5 model
model = torch.hub.load('backend/yolov5', 'custom', path='backend/best.pt', source='local')

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    results = model(image)
    results.render()

    # Convert to base64 image
    rendered_img = Image.fromarray(results.ims[0])
    buffered = io.BytesIO()
    rendered_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content={"image": img_str})
    
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
