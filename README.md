# 🧠 Tweet Sentiment Classifier

A machine learning web app that classifies tweets as **Positive** or **Negative** using Natural Language Processing (NLP) techniques and a Logistic Regression model. Built using Python, Scikit-learn, and deployed via Streamlit.

---

## 🚀 Live Demo

👉 [Launch the App](https://ml-sentiment-analysis13.streamlit.app/)  
⚡ Real-time prediction based on user-inputted tweet text.

---

## 📌 Features

- 🧹 Text cleaning, stemming, and stopword removal using NLTK
- 📊 TF-IDF vectorization for feature extraction
- 🤖 Logistic Regression model for binary classification
- 🌐 Interactive web interface with Streamlit
- ✅ Accuracy: **Train – 80%**, **Test – 77%**

---

## 📂 File Structure

ml-sentiment-analysis

├── app.py # Streamlit web app

├── logistic.pkl # Trained Logistic Regression model

├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer

├── requirements.txt # Python dependencies

└── README.md # Project documentation



---

## 📦 Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/Lakhan-Bhutiya/ml-sentiment-analysis.git
cd ml-sentiment-analysis
pip install -r requirements.txt

⚙️ Model & Preprocessing
Preprocessed using regex, stemming (PorterStemmer), and NLTK stopwords

Features extracted via TfidfVectorizer

Classification model: LogisticRegression()



