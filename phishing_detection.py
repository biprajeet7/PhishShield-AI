import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from nltk.corpus import stopwords
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import nltk
# Download NLTK data
nltk.download('stopwords')
# Load stopwords
STOPWORDS = set(stopwords.words("english"))
def clean_email_body(email_body):
    email_body = re.sub(r'[^a-zA-Z\s]', '', email_body)
    email_body = ' '.join([word.lower() for word in email_body.split() if word.lower() not in STOPWORDS])
    return email_body
def extract_url_features(email_body):
    urls = re.findall(r'(https?://[^\s]+)', email_body)
    features = []
    
    for url in urls:
        parsed_url = urlparse(url)
        domain_length = len(parsed_url.netloc)
        path_length = len(parsed_url.path)
        protocol = 1 if parsed_url.scheme in ['http', 'https'] else 0
        
        features.extend([domain_length, path_length, protocol])
        
        try:
            response = requests.get(url, timeout=3)
            features.append(response.status_code)
        except:
            features.append(0)
    
    return np.mean(features) if features else 0
def train_model(data):
    data['cleaned_body'] = data['email_body'].apply(clean_email_body)
    data['url_features'] = data['email_body'].apply(extract_url_features)
    X_train, X_test, y_train, y_test = train_test_split(
        data[['cleaned_body', 'url_features']], data['label'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_body = vectorizer.fit_transform(X_train['cleaned_body'])
    X_test_body = vectorizer.transform(X_test['cleaned_body'])
    
    model = MultinomialNB()
    model.fit(X_train_body, y_train)
    
    accuracy = model.score(X_test_body, y_test)
    
    joblib.dump((vectorizer, model), 'phishing_model.pkl')
    return accuracy
def predict_phishing(model, email_body):
    vectorizer, classifier = model
    cleaned_body = clean_email_body(email_body)
    url_features = extract_url_features(email_body)
    
    X_transformed = vectorizer.transform([cleaned_body])
    prediction = classifier.predict(X_transformed)
    
    return 'Phishing' if prediction[0] == 1 else 'Safe'
def main():
    st.set_page_config(page_title="Phishing Detection", layout="wide")
    
    st.sidebar.image("logo.JPEG", width=200)
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select an option:", ["Phishing Detection", "Train Model"])
    
    if option == "Phishing Detection":
        st.title("🚨 PhishShield AI")
        st.write("Use this tool to detect phishing emails using AI.")
        
        try:
            model = joblib.load('phishing_model.pkl')
            st.success("✅ Model loaded successfully!")
        except:
            st.warning("⚠ No trained model found. Please train the model first.")
        
        email_input = st.text_area("Enter Email Content:", height=200)
        
        if st.button("Detect Phishing", use_container_width=True):
            if email_input:
                result = predict_phishing(model, email_input)
                st.subheader(f"Prediction: {result}")
            else:
                st.error("Please enter email content for analysis.")
    
    elif option == "Train Model":
        st.title("📊 Train Phishing Detection Model")
        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'email_body' in data.columns and 'label' in data.columns:
                st.write("Training the model with uploaded data...")
                accuracy = train_model(data)
                st.success(f"🎯 Model trained with {accuracy * 100:.2f}% accuracy!")
            else:
                st.error("Invalid dataset format. Ensure 'email_body' and 'label' columns are present.")
if __name__ == '__main__':
    main()


