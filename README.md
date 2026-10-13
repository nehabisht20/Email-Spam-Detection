# InboxShield — Email Spam Detection

An NLP-based email spam detection project using machine learning, with an interactive Streamlit interface and an optional FastAPI endpoint.

## Features
- Spam classification using Logistic Regression
- Bag of Words and TF-IDF comparison
- Model evaluation using accuracy, precision, recall, and F1-score
- N-gram analysis with Laplace smoothing
- POS tagging, lemmatization, and Named Entity Recognition
- Sentiment analysis using VADER
- Word2Vec embeddings and word similarity

## Tech Stack
Python, Streamlit, scikit-learn, spaCy, NLTK, Gensim, FastAPI.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To run the optional API:

```bash
uvicorn api:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`

## Dataset
Uses 100 labeled synthetic emails: 50 spam and 50 not spam. Intended for educational purposes.
