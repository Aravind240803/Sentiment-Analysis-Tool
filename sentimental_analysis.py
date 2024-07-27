# sentiment_analysis.py

import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load data
def load_data(filepath):
    print("Loading data from the file...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows of data.")
    return df

# Preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return ' '.join(filtered_tokens)

# Apply preprocessing to the dataset
def preprocess_data(df, text_column):
    print("Preprocessing the text data...")
    df[text_column] = df[text_column].apply(preprocess_text)
    print("Text data preprocessing completed.")
    return df

# Feature extraction using TF-IDF
def extract_features(df, text_column):
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[text_column]).toarray()
    print("Feature extraction completed.")
    return X, vectorizer

# Train and evaluate model
def train_evaluate_model(X, y):
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f'Accuracy: {accuracy:.2f} - The percentage of correct predictions.')
    print(f'Precision: {precision:.2f} - The ability of the model to not label a negative sample as positive.')
    print(f'Recall: {recall:.2f} - The ability of the model to find all the positive samples.')
    print(f'F1-score: {f1:.2f} - The balance between precision and recall.')
    
    visualize_results(accuracy, precision, recall, f1)
    
    return model

# Visualize results
def visualize_results(accuracy, precision, recall, f1):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = [accuracy, precision, recall, f1]
    
    plt.bar(metrics, scores, color=['blue', 'orange', 'green', 'red'])
    plt.ylim([0, 1])
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Model Evaluation Metrics')
    plt.show()
