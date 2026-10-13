# Optional REST API
# This file is not required to run the Streamlit app.
# It demonstrates the REST-API part of the syllabus.

from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "emails.csv")
df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X = vectorizer.fit_transform(df["text"])
model = LogisticRegression(max_iter=2000)
model.fit(X, df["label"])

app = FastAPI(title="InboxShield Spam Detection API")

class EmailRequest(BaseModel):
    subject: str = ""
    body: str = ""

@app.get("/")
def home():
    return {"message": "InboxShield API is running"}

@app.post("/predict")
def predict(email: EmailRequest):
    text = f"{email.subject} {email.body}".strip()
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    spam_probability = float(probabilities[classes.index("spam")])
    return {
        "prediction": prediction,
        "spam_probability": spam_probability
    }

# Run separately with:
# uvicorn api:app --reload
