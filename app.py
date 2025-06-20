import pickle as pk
import streamlit as st

# Load model and vectorizer
model = pk.load(open('model.pk1', 'rb'))
vectorizer = pk.load(open('scaler.pk1', 'rb'))  # this must be the same vectorizer used during training

# Streamlit UI
st.title("Movie Sentiment Classifier")
review = st.text_input("Enter movie review:")

if st.button("Predict"):
    review_vector = vectorizer.transform([review])  # very important: transform using same vectorizer
    result = model.predict(review_vector)
    sentiment = "Negative 😞" if result[0] == 1 else "Positive 😊"
    st.success(f"Sentiment: {sentiment}")
