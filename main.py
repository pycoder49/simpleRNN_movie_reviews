# Loading necessary libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

# load imdb dataset and word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# load the saved model
model = load_model("simple_rnn_imdb_model.h5")

# function to decode reviews
def decode_review(text):
    return " ".join([reverse_word_index.get(word - 3, "?") for word in text])

# function to preprocess user input
def preprocess_review(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# prediction function
def predict_review_sentiment(review):
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]


# streamlit app
import streamlit as st

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")

# getting user input
user_review = st.text_area("Movie Review:")

if st.button("Predict Sentiment"):
    preprocessed_input = preprocess_review(user_review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    confidence = prediction[0][0]
    st.write(f"Sentiment: **{sentiment}** (Confidence: {confidence:.4f})")
else:
    st.write("Please enter a review and click 'Predict Sentiment' to see the result.")