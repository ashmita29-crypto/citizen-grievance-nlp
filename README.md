# Citizen Grievance Classification & Sentiment Analysis System

## Project Overview
An AI-powered NLP system that automatically classifies citizen
complaints into government departments and detects urgency level,
built as part of the Infotact DS/ML Internship Program.

## Problem Statement
Government departments receive thousands of unstructured citizen
complaints. Manual routing is slow and misses critical emergencies.
This system automates routing and priority detection.

## Features
- Department classification (Water, Electricity, Roads,
  Sanitation, Transport)
- Sentiment analysis (Positive, Neutral, Negative, Urgent/Critical)
- Mathematical priority scoring (0.0 to 1.0)
- REST API endpoint via FastAPI

## Tech Stack
- Python 3.11
- NLTK (text preprocessing)
- Scikit-Learn (TF-IDF + Logistic Regression)
- FastAPI + Uvicorn (model serving)
- Matplotlib / Seaborn (visualization)

## Project Structure
citizen-grievance-nlp/
├── notebooks/         # Jupyter notebooks (01 to 06)
├── src/               # Saved model files (.pkl)
├── api/               # FastAPI application
│   └── main.py
├── data/
│   ├── raw/           # Original dataset
│   └── processed/     # Cleaned data + charts
└── requirements.txt

## How to Run Locally
1. Clone this repository
2. Create virtual environment: python -m venv venv
3. Activate: venvScriptsactivate
4. Install dependencies: pip install -r requirements.txt
5. Start API: uvicorn api.main:app --reload
6. Visit: http://127.0.0.1:8000/docs

## API Usage
POST /predict
{
  "complaint_text": "No water supply since 2 days"
}

Response:
{
  "department": "Water",
  "dept_confidence": 0.92,
  "sentiment": "Negative",
  "sent_confidence": 0.81,
  "priority_score": 0.61,
  "priority_level": "High"
}

## Model Performance
- Department Classifier Accuracy: 65%
- Department Macro F1: 0.6407
- Sentiment Classifier Accuracy: 75%
- Sentiment Macro F1: 0.7760

## Engineering Roadmap Completed
- Week 1: Data collection, text cleaning, EDA
- Week 2: TF-IDF vectorization, department classifier
- Week 3: Sentiment analysis, urgency scoring
- Week 4: Dual-output FastAPI, final evaluation
