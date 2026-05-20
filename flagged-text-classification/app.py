from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI app
app = FastAPI()

# Load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Label mapping
label_map = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "love",
    5: "surprise"
}

# Input schema
class TextInput(BaseModel):
    text: str

# Home route
@app.get("/")
def home():
    return {
        "message": "Emotion Classification API Running"
    }

# Prediction route
@app.post("/predict")
def predict(data: TextInput):

    # Transform input text
    transformed_text = vectorizer.transform([data.text])

    # Predict emotion
    prediction = model.predict(transformed_text)

    # Convert prediction to integer
    predicted_label = int(prediction[0])

    # Convert number to emotion label
    emotion = label_map[predicted_label]

    # Return response
    return {
        "text": data.text,
        "prediction": emotion
    }
