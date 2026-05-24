from fastapi import FastAPI
from pydantic import BaseModel

import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# Create FastAPI app
# =========================

app = FastAPI(
    title="Citizen Grievance Classification API",
    description="Classifies citizen complaints into departments",
    version="1.0.0"
)

# =========================
# Load saved model + vectorizer
# =========================

with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("src/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =========================
# NLP setup
# =========================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

# =========================
# Request schema
# =========================

class ComplaintRequest(BaseModel):
    complaint_text: str

# =========================
# Response schema
# =========================

class PredictionResponse(BaseModel):
    original_text: str
    department: str
    confidence: float

# =========================
# Text preprocessing
# =========================

def preprocess(text: str) -> str:

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    tokens = text.split()

    tokens = [
        w for w in tokens
        if w not in stop_words and len(w) > 2
    ]

    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
    ]

    return ' '.join(tokens)

# =========================
# Health check route
# =========================

@app.get("/")
def root():

    return {
        "message": "Grievance API is running!"
    }

# =========================
# Prediction route
# =========================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: ComplaintRequest):

    cleaned = preprocess(request.complaint_text)

    vectorized = vectorizer.transform([cleaned])

    department = model.predict(vectorized)[0]

    probabilities = model.predict_proba(vectorized)[0]

    confidence = round(float(max(probabilities)), 4)

    return PredictionResponse(
        original_text=request.complaint_text,
        department=department,
        confidence=confidence
    )