from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import os
import shutil
import zipfile
import uuid
from pathlib import Path
from ml.hyperparameters import SystemSpecs, HyperparameterRecommendation, recommend_hyperparameters
from ml.model import train_model_sync
import threading
app = FastAPI(title="Hyperparameter Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    upload_id = str(uuid.uuid4())
    upload_path = TEMP_UPLOAD_DIR / f"{upload_id}.zip"
    extract_path = TEMP_UPLOAD_DIR / upload_id
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    extract_path.mkdir(exist_ok=True)
    with zipfile.ZipFile(upload_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        
    # Analyze the dataset structure
    classes = [d.name for d in extract_path.iterdir() if d.is_dir()]
    num_classes = len(classes)
    
    return {"message": "Dataset uploaded successfully", "dataset_id": upload_id, "num_classes": num_classes, "classes": classes}

@app.post("/recommend", response_model=HyperparameterRecommendation)
async def recommend(specs: SystemSpecs):
    return recommend_hyperparameters(specs)

@app.websocket("/train_ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    config_text = await websocket.receive_text()
    config = json.loads(config_text)
    
    dataset_name = config.get("dataset_name", "mnist")
    lr = float(config.get("learning_rate", 0.001))
    epochs = int(config.get("epochs", 5))
    batch_size = int(config.get("batch_size", 32))
    optimizer_name = config.get("optimizer_name", "adam")
    
    async_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    # Run the Keras fit synchronously in a separate thread so it doesn't block the API
    thread = threading.Thread(
        target=train_model_sync,
        args=(dataset_name, lr, optimizer_name, epochs, batch_size, async_queue, loop)
    )
    thread.start()
    
    try:
        while True:
            # Wait for data from the Keras callback (yielded into queue)
            data = await async_queue.get()
            
            # Send data to frontend
            await websocket.send_json(data)
            
            if data.get("type") in ["training_complete", "error"]:
                break
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()
