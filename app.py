import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy.special import expit

# Load the model and vectorizer using joblib
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news headline below to check if it's real or fake.")

# User input
headline = st.text_input("News Headline", "")

if st.button("Check"):
    if headline:
        # Transform input text
        input_vector = vectorizer.transform([headline])
        
        # Predict
        prediction = model.predict(input_vector)[0]
        confidence = expit(model.decision_function(input_vector)) * 100  
        # Display result
        if prediction == 'FAKE':
            st.error(f"🚨 Fake News! (Confidence: {confidence[0]:.2f}%)")
        else:
            st.success(f"✅ Real News!Confidence: {confidence[0]:.2f}%)")

    else:
        st.warning("Please enter a valid headline.")

input_vector = vectorizer.transform([headline])
prediction = model.predict(input_vector)[0]
print(prediction)
print({word: value for word, value in zip(vectorizer.get_feature_names_out(), vectorizer.transform([headline]).toarray()[0]) if value > 0})

