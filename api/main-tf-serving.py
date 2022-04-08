# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 01:32:51 2021

@author: vidhy
"""

from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# uvicorn main:app --reload

# install docker
# docker pull tensorflow/serving
# cd D:/deep-learning/
# docker run -t --rm -p 8501:8501 -v D:/deep-learning/potato-disease-classification:/potato-disease-classification tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease-classification/models.config


endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


@app.get("/ping")
async def ping():
    return "Hello, I'm alive"

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


def read_file_as_image(data) -> np.ndarray:
    # bytes = await file.read()

    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    prediction = response.json()["predictions"][0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
        }

if __name__ == "__main__":
   uvicorn.run(app, host='localhost', port=8000)
