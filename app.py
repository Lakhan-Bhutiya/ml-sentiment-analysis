import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')


with open('logistic.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text preprocessing function
port_stem = PorterStemmer()

def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)


st.title("Tweet Sentiment Classifier")
st.write("Enter a tweet to predict whether it's positive or negative.")

user_input = st.text_area("Tweet text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        processed = preprocess_text(user_input)
        vect_input = vectorizer.transform([processed]).toarray()
        prediction = model.predict(vect_input)[0]

        if prediction == 1:
            st.success("✅ Positive Tweet")
        else:
            st.error("❌ Negative Tweet")
