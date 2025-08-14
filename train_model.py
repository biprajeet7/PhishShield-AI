import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from urllib.parse import urlparse
import requests
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

# Load dataset
print("Loading dataset...")
df = pd.read_csv("emails.csv")

# Function to clean and preprocess email body
def clean_email_body(email_body):
    email_body = re.sub(r'[^a-zA-Z\s]', '', email_body)  # Remove non-alphabetic characters
    email_body = ' '.join([word.lower() for word in email_body.split() if word.lower() not in STOPWORDS])
    return email_body

# Extract URL features (e.g., domain length, path length, HTTP status)
def extract_url_features(email_body):
    urls = re.findall(r'(https?://[^\s]+)', email_body)
    features = []
    
    for url in urls:
        parsed_url = urlparse(url)
        domain_length = len(parsed_url.netloc)  # Domain length
        path_length = len(parsed_url.path)  # Path length
        protocol = 1 if parsed_url.scheme in ['http', 'https'] else 0
        
        # Fetch URL status (if reachable)
        try:
            response = requests.get(url, timeout=3)
            status_code = response.status_code
        except:
            status_code = 0  # Not reachable
        
        features.extend([domain_length, path_length, protocol, status_code])

    return np.mean(features) if features else 0  # Return mean of extracted features

# Apply cleaning and feature extraction
print("Cleaning email body and extracting URL features...")
df['cleaned_body'] = df['email_body'].apply(clean_email_body)
df['url_features'] = df['email_body'].apply(extract_url_features)

# Prepare training data
X_text = df['cleaned_body']
X_url_features = df['url_features'].values.reshape(-1, 1)  # Reshape for concatenation
y = df['label']

# Train-test split
print("Splitting dataset into training and testing sets...")
X_train_text, X_test_text, X_train_url, X_test_url, y_train, y_test = train_test_split(
    X_text, X_url_features, y, test_size=0.2, random_state=42
)

# Text feature extraction using TF-IDF
print("Vectorizing text data...")
vectorizer = TfidfVectorizer()
X_train_text_vectorized = vectorizer.fit_transform(X_train_text)
X_test_text_vectorized = vectorizer.transform(X_test_text)

# Combine text and URL features
print("Combining text and URL features...")
X_train_combined = np.hstack((X_train_text_vectorized.toarray(), X_train_url))
X_test_combined = np.hstack((X_test_text_vectorized.toarray(), X_test_url))

# Train Naive Bayes model
print("Training the model...")
model = MultinomialNB()
model.fit(X_train_combined, y_train)

# Save trained model and vectorizer
print("Saving the trained model...")
joblib.dump((model, vectorizer), "phishing_model.pkl")

print("Model retrained and saved as 'phishing_model.pkl'.")


