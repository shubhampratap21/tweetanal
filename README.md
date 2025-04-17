# Public Sentiment Analyzer

A web-based sentiment analysis application that detects the mood behind your input text and responds with motivational or reflective quotes based on the sentiment. Built using Hugging Face Transformers and Streamlit.

---

## Features

- Real-time Sentiment Detection using `cardiffnlp/twitter-roberta-base-sentiment`
- Mood Diary to track your past entries and moods
- Motivational Quotes tailored to detected sentiment (Positive, Neutral, Negative)
- Visual Evaluations:
  - Actual vs Predicted Sentiment Graph
  - Word Frequency Analysis
- Classification Report for model performance overview
- Emoji Feedback System for user interaction

---

## Tech Stack

- Frontend: Streamlit
- Model: [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- Backend: Transformers Pipeline
- Language: Python

---

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/public-sentiment-analyzer.git
   cd public-sentiment-analyzer
