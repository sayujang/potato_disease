import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import tensorflow as tf
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
classes = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data: bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
@app.get("/here")
async def b():
    return "sup dwag"

@app.post("/predict")  
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        model = tf.keras.models.load_model(r"D:\codeprojects\potatoclassifier\models\1")
        imag_batc = np.expand_dims(image, 0)
        prediction = model.predict(imag_batc)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = classes[predicted_class_index]
        return {"class": predicted_class}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
