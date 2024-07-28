# Sentiment Analysis Tool

The Sentiment Analysis Tool is a user-friendly application designed to analyze text data and determine the sentiment expressed. By leveraging machine learning techniques, it provides insights into the emotional tone of the text, helping users understand and act on the sentiment.

## Features

- **Data Loading**: Easily load text data from CSV files for analysis.
- **Text Preprocessing**: Convert text to lowercase, remove punctuation, and eliminate stop words for cleaner data.
- **Feature Extraction**: Use TF-IDF vectorization to transform text into numerical features.
- **Model Training and Evaluation**: Train a Logistic Regression model and evaluate its performance with metrics like accuracy, precision, recall, and F1-score.
- **Confusion Matrix Visualization**: Visualize the confusion matrix to understand the model's performance in terms of true positives, false positives, true negatives, and false negatives.

## Install dependencies

pip install -r requirements.txt

## Usage

- Run the GUI application: python sentiment_analysis_gui.py
- Load Data: Use the interface to load the dataset from a CSV file.
- Preprocess Data: Enter the name of the text column and preprocess the text data.
- Train and Evaluate Model: Enter the name of the label column, train the model, and visualize the confusion matrix.

## Result

The tool processes the text data, trains a Logistic Regression model, and evaluates its performance. The results, including accuracy, precision, recall, and F1-score, are displayed along with a confusion matrix visualization.

## Future Improvements

Model Enhancement: Further fine-tuning with diverse datasets to improve accuracy.
Feature Expansion: Adding more NLP features such as keyword matching and sentiment scoring.
User Interface Improvements: Enhancing the UI for a better user experience.




