# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from predict import load_model, predict
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str

model = load_model("vietnamese_best.pth.tar")

@app.post("/recognize")
async def recognize(data: ImageData):
    _, base64_data = data.image.split(",")
    image = Image.open(BytesIO(base64.b64decode(base64_data))).convert("RGB")
    image = np.array(image)

    # plt.figure(figsize=(15,2))
    # plt.imshow(image)
    # plt.show()

    text = predict(model, image)
    return {"text": text}
