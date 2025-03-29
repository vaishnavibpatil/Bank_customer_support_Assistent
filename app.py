import streamlit as st
from predict import predict_intent

st.title("🤖 Banking Intent Classifier")

user_query = st.text_input("Enter your banking query:", "")

if st.button("Predict Intent"):
    if user_query.strip():
        predicted_intent = predict_intent(user_query)
        st.write(f"**Predicted Intent:** {predicted_intent} ✅")  # Show intent name
    else:
        st.warning("⚠️ Please enter a query to classify.")
