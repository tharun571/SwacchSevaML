from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import asyncio

app = FastAPI()

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def predict_image(image):
    np.set_printoptions(suppress=True)

    model = tensorflow.keras.models.load_model('keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(image)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    return prediction    

@app.get('/index')
def hello():
    return "helolo"

@app.post('/api/predict')
async def predictImage(file: UploadFile = File(...)):
    image = read_image(file)
    prediction = predict_image(image)   
    
    return prediction 

if __name__ == "__main__":
    uvicorn.run(app, port=8081, host='0.0.0.0')

