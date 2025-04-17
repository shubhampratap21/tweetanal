from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import random

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Label map
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# Initialize sentiment analyzer
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

# Extended Quotes
QUOTES = {
    "positive": [
        "Keep your face always toward the sunshine—and shadows will fall behind you.",
        "Every day may not be good... but there’s something good in every day.",
        "Believe you can and you're halfway there.",
        "Positive thinking evokes more energy, more initiative, more happiness.",
        "The best way to predict the future is to create it.",
        "Start each day with a positive thought and a grateful heart.",
        "Your vibe attracts your tribe.",
        "Happiness is not by chance, but by choice.",
        "You are capable of amazing things.",
        "Choose to be optimistic—it feels better."
    ],
    "negative": [
        "Even the darkest night will end and the sun will rise.",
        "Tough times never last, but tough people do.",
        "It's okay to not be okay. Just don't give up.",
        "This too shall pass.",
        "You don’t have to control your thoughts. You just have to stop letting them control you.",
        "Cry. Forgive. Learn. Move on. Let your tears water the seeds of your future happiness.",
        "Healing takes time, and that’s okay.",
        "Sometimes, just surviving another day is courage enough.",
        "Your current situation is not your final destination.",
        "The comeback is always stronger than the setback."
    ],
    "neutral": [
        "Stay focused and keep moving forward.",
        "Let things flow naturally forward in whatever way they like.",
        "Every moment is a fresh beginning.",
        "Silence is sometimes the best answer.",
        "Not every situation needs a reaction.",
        "Be still and calm, even in chaos.",
        "What you do makes a difference, and you have to decide what kind of difference you want to make.",
        "Sometimes you just need to pause and reset.",
        "Balance is not something you find, it’s something you create.",
        "Everything comes to you at the right time."
    ]
}

# Layout
st.set_page_config(layout="wide")
st.title("🧠 Public Sentiment Analyzer")

# Mood Diary Init
if 'mood_log' not in st.session_state:
    st.session_state.mood_log = []

# Input section
col1, col2 = st.columns([100, 1])

with col1:
    user_input = st.text_input("💬 How are you feeling today?")

    if user_input:
        result = sentiment_analyzer(user_input[:512])[0]
        sentiment = label_map.get(result['label'], "neutral")
        confidence = result['score']

        # Store in diary
        st.session_state.mood_log.append((user_input, sentiment.upper(), f"{confidence*100:.1f}%"))

        # Show results
        st.subheader("Analysis Results")
        st.success(f"**Detected Mood**: {sentiment.upper()} ({confidence*100:.1f}% confidence)")

        # Quote
        quote = random.choice(QUOTES.get(sentiment, QUOTES['neutral']))
        st.info(f"**Suggested Quote**: {quote}")

        # Emoji Feedback
        st.write("Was this helpful?")
        colA, colB = st.columns([5,20])
        with colA:
            if st.button("👍 Yes"):
                st.success("Thanks for your feedback!")
        with colB:
            if st.button("👎 No"):
                st.warning("We'll try to improve it!")

        # Mood Diary
        with st.expander("📔 Mood Diary"):
            for idx, (text, mood, score) in enumerate(reversed(st.session_state.mood_log), 1):
                st.markdown(f"**{idx}.** _\"{text}\"_ → **{mood}** ({score} confidence)")


# --- Evaluations Section ---
st.header("📂 Evaluations")

# First Row: Visuals
col3, col4 = st.columns(2)

with col3:
    st.subheader("Actual vs Predicted Sentiment")
    st.image("eval_graph.png", caption="Left: Actual Sentiment | Right: Predicted Sentiment", use_column_width=True)

with col4:
    st.subheader("Word Frequency in Tweets")
    st.image("freq.png", caption="Most Frequent Words in Tweets", use_column_width=True)

# Second Row: Classification Report
st.subheader("📄 Classification Report")


# Metrics in table format
st.markdown("""
| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Positive  | 0.63      | 0.59   | 0.61     | 14651   |
| Negative  | 0.72      | 0.75   | 0.73     | 20331   |
| Neutral   | 0.00      | 0.00   | 0.00     | 0       |
| **Accuracy** |         |        | **0.68** | 34982   |
| **Macro Avg** | 0.45  | 0.45   | 0.45     | 34982   |
| **Weighted Avg** | 0.68 | 0.68 | 0.68     | 34982   |
""")

