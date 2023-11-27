from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
from pathlib import Path
from src.Textclassification.config.configuration import ConfigurationManager  # Adjust the import based on your project structure
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
import os
import uvicorn

app = FastAPI()

class Item(BaseModel):
    description: str

# Load configuration
config_manager = ConfigurationManager()
model_trainer_config = config_manager.get_data_transformation_config()

# Load model and tokenizer
save_directory = model_trainer_config.data_path1
save_directory1 = model_trainer_config.data_path2

tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_directory1)
model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# API endpoint for prediction
@app.post("/predict")
async def predict(item: Item):
    try:
        # Tokenize and predict
        predict_input = tokenizer_fine_tuned.encode(
            item.description,
            truncation=True,
            padding=True,
            return_tensors='tf'
        )

        output = model_fine_tuned(predict_input)[0]
        prediction_value = tf.argmax(output, axis=1).numpy()[0]

        # Mapping between numerical labels and classes
        label_mapping = {0: "Business", 1: "Entertainment", 2: "Politics", 3: "Sports", 4: "Tech"}

        # Convert numerical result to class
        predicted_class = label_mapping[prediction_value]

        return {"predicted_class": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
