from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = None

# Lazy load model
def get_model():
    global model
    if model is None:
        model = YOLO("best.pt")
        model.to("cpu")   # ensure CPU inference
    return model


@app.get("/")
def home():
    return {"message": "Apple Disease Detection API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Resize image to reduce memory usage
    image = image.resize((224, 224))

    model = get_model()

    results = model(image, device="cpu")

    probs = results[0].probs
    class_id = probs.top1
    confidence = float(probs.top1conf)

    class_name = model.names[class_id]

    return {
        "class": class_name,
        "confidence": confidence
    }


@app.post("/predict_multiple")
async def predict_multiple(files: list[UploadFile] = File(...)):

    model = get_model()

    predictions = []

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        results = model(image, device="cpu")

        probs = results[0].probs
        class_id = probs.top1
        confidence = float(probs.top1conf)

        class_name = model.names[class_id]

        predictions.append({
            "image": file.filename,
            "class": class_name,
            "confidence": confidence
        })

    return predictions