from fastapi import FastAPI
from pydantic import BaseModel
import pickle, re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# App setup 
app = FastAPI(
    title="Citizen Grievance Classification API",
    description="Classifies complaints into departments and detects urgency",
    version="2.0.0"
)

# Load all 4 model files at startup 
# Department model
with open("src/model.pkl", "rb") as f:
    dept_model = pickle.load(f)
with open("src/vectorizer.pkl", "rb") as f:
    dept_vectorizer = pickle.load(f)

# Sentiment model
with open("src/sentiment_model.pkl", "rb") as f:
    sent_model = pickle.load(f)
with open("src/sentiment_vectorizer.pkl", "rb") as f:
    sent_vectorizer = pickle.load(f)

# NLP tools
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

# Priority score mapping
PRIORITY_MAP = {
    "Positive":        1,
    "Neutral":         2,
    "Negative":        3,
    "Urgent/Critical": 4
}

# Helper functions
def preprocess(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"[^a-zs]", "", text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)

def priority_score(sentiment: str, confidence: float) -> float:
    base  = PRIORITY_MAP.get(sentiment, 1)
    score = (base / 4) * confidence
    return round(score, 4)

# Request / Response schemas 
class ComplaintRequest(BaseModel):
    complaint_text: str

class PredictionResponse(BaseModel):
    original_text:  str
    cleaned_text:   str
    department:     str
    dept_confidence: float
    sentiment:      str
    sent_confidence: float
    priority_score: float
    priority_level: str   # human-readable label
    
# Endpoints 
@app.get("/")
def root():
    return {
        "message": "Citizen Grievance API v2.0 is running",
        "endpoints": ["/predict", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ComplaintRequest):
    cleaned = preprocess(request.complaint_text)

    # Department prediction
    d_vec    = dept_vectorizer.transform([cleaned])
    dept     = dept_model.predict(d_vec)[0]
    d_proba  = dept_model.predict_proba(d_vec).max()

    # Sentiment prediction
    s_vec    = sent_vectorizer.transform([cleaned])
    sent     = sent_model.predict(s_vec)[0]
    s_proba  = sent_model.predict_proba(s_vec).max()
    
    # Priority score and label
    p_score  = priority_score(sent, float(s_proba))
    p_levels = {
        (0.0, 0.25): "Low",
        (0.25, 0.50): "Medium",
        (0.50, 0.75): "High",
        (0.75, 1.01): "Critical"
    }
    p_label = next(
        label for (lo, hi), label in p_levels.items()
        if lo <= p_score < hi
    )

    return PredictionResponse(
        original_text   = request.complaint_text,
        cleaned_text    = cleaned,
        department      = dept,
        dept_confidence = round(float(d_proba), 4),
        sentiment       = sent,
        sent_confidence = round(float(s_proba), 4),
        priority_score  = p_score,
        priority_level  = p_label
    )
    
    