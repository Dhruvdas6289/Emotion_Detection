from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from collections import Counter
from functools import lru_cache
import os

app = Flask(__name__)

# Lazy loading of model to reduce memory at startup
@lru_cache(maxsize=1)
def get_model():
    return load_model('gru_emotion_model_v3.h5')

@lru_cache(maxsize=1)
def get_scaler():
    return joblib.load('scaler.pkl')

@lru_cache(maxsize=1)
def get_encoder():
    return joblib.load('encoder.pkl')

# Load train labels only once
train_labels = None
if os.path.exists('emotions.csv'):
    train_df = pd.read_csv('emotions.csv', nrows=10)
    if 'label' not in train_df.columns:
        raise KeyError("The 'emotions.csv' file must contain a 'label' column.")
    train_labels = train_df['label'].values
else:
    raise FileNotFoundError("emotions.csv not found.")

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    actual_vs_pred = []
    labels = []
    counts = []

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            try:
                # Read first 10 rows for performance
                df = pd.read_csv(file, nrows=10)
                X = df.loc[:, 'fft_0_b':'fft_749_b'].values

                scaler = get_scaler()
                encoder = get_encoder()
                model = get_model()

                X_scaled = scaler.transform(X)
                X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

                # Predict
                y_pred = model.predict(X_input)
                predicted_labels = encoder.inverse_transform(np.argmax(y_pred, axis=1))
                predictions = predicted_labels.tolist()

                # Actual labels for comparison
                actual_labels = train_labels[:len(predictions)]
                actual_vs_pred = list(zip(actual_labels, predictions))

                # Count distribution
                label_counts = Counter(predicted_labels)
                total = sum(label_counts.values())
                labels = list(label_counts.keys())
                counts = [round((label_counts[l] / total) * 100, 2) for l in labels]

            except Exception as e:
                predictions = [f"Error: {str(e)}"]

    return render_template('index.html',
                           predictions=predictions,
                           actual_vs_pred=actual_vs_pred,
                           labels=labels,
                           counts=counts)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
