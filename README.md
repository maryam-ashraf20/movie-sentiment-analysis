# Movie Sentiment Analysis

## Overview
A machine learning application that classifies English movie reviews as positive or negative using Natural Language Processing (NLP).

## 🚀 Live Demo

Try the deployed application here:

**[🎬 Movie Sentiment Analyzer — Live App](https://movie-sentiment-analysis-p6qoytgvdmrgcseyup9brg.streamlit.app/)**

Enter an English movie review and see whether the model predicts a positive or negative sentiment.
## Features
- Text cleaning and tokenization
- Lemmatization using NLTK
- TF-IDF text vectorization
- Linear Support Vector Classification (LinearSVC)
- Interactive web interface built with Streamlit

## Model Performance
- Accuracy: 90.84%
- F1-score: 90.95%

These metrics were measured on the project's held-out test set. Performance on new reviews may vary.

## Tech Stack
- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF
- LinearSVC
- Streamlit
- Joblib

## Run Locally

1. Clone the repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Run the application:

   `streamlit run app.py`

## Project Structure

- `app.py` — Streamlit application
- `preprocessing.py` — Text preprocessing functions
- `sentiment_model.pkl` — Trained classification model
- `tfidf_vectorizer.pkl` — Fitted TF-IDF vectorizer
- `requirements.txt` — Project dependencies

## Limitations
The model may misclassify unfamiliar phrases, sarcasm, and reviews whose wording differs from the training data.

## Future Improvements
- Evaluate additional models
- Analyze misclassified reviews
- Improve preprocessing and error analysis
- Deploy the application online
