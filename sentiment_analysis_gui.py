# sentiment_analysis_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox
from sentimental_analysis import load_data, preprocess_data, extract_features, train_evaluate_model

# GUI functions
def load_file():
    filepath = filedialog.askopenfilename(filetypes=[("IMDB Dataset", "*.csv")])
    if filepath:
        global df
        df = load_data(filepath)
        messagebox.showinfo("Info", f"Data loaded from {filepath}")

def preprocess():
    global df
    text_column = text_column_entry.get()
    if text_column in df.columns:
        df = preprocess_data(df, text_column)
        messagebox.showinfo("Info", "Text data preprocessing completed")
    else:
        messagebox.showerror("Error", f"Column {text_column} not found in the dataset")

def extract_features_and_train():
    global df
    text_column = text_column_entry.get()
    label_column = label_column_entry.get()
    if text_column in df.columns and label_column in df.columns:
        X, vectorizer = extract_features(df, text_column)
        y = df[label_column].apply(lambda x: 1 if x == 'positive' else 0)  # Convert labels to binary
        train_evaluate_model(X, y)
    else:
        messagebox.showerror("Error", "Text or Label column not found in the dataset")

# Create GUI
root = tk.Tk()
root.title("Sentiment Analysis Tool")

load_button = tk.Button(root, text="Load Data", command=load_file)
load_button.pack()

text_column_label = tk.Label(root, text="Text Column:")
text_column_label.pack()
text_column_entry = tk.Entry(root)
text_column_entry.pack()

label_column_label = tk.Label(root, text="Label Column:")
label_column_label.pack()
label_column_entry = tk.Entry(root)
label_column_entry.pack()

preprocess_button = tk.Button(root, text="Preprocess Data", command=preprocess)
preprocess_button.pack()

train_button = tk.Button(root, text="Train and Evaluate Model", command=extract_features_and_train)
train_button.pack()

root.mainloop()
