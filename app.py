from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from collections import Counter

app = Flask(__name__)

# Load model and preprocessing tools
model = load_model('gru_emotion_model_v3.h5')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Load the original dataset with labels
train_df = pd.read_csv('emotions.csv')
if 'label' not in train_df.columns:
    raise KeyError("The 'emotions.csv' file must contain a 'label' column.")

train_labels = train_df['label'].values

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
                # Read uploaded CSV and extract FFT features
                df = pd.read_csv(file)
                X = df.loc[:, 'fft_0_b':'fft_749_b'].values
                X_scaled = scaler.transform(X)
                X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

                # Predict with GRU model
                y_pred = model.predict(X_input)
                predicted_labels_all = encoder.inverse_transform(np.argmax(y_pred, axis=1))

                # Limit to first 10 predictions
                top10_predicted = predicted_labels_all[:10]
                predictions = top10_predicted.tolist()

                # Actual labels for comparison (optional)
                actual_labels = train_labels[:10]
                actual_vs_pred = list(zip(actual_labels, predictions))

                # Count distribution of top 10 predicted values
                label_counts = Counter(top10_predicted)
                total_top10 = sum(label_counts.values())
                labels = list(label_counts.keys())
                counts = [round((label_counts[l] / total_top10) * 100, 2) for l in labels]

            except Exception as e:
                predictions = [f"Error: {str(e)}"]
                actual_vs_pred = []
                labels = []
                counts = []

    return render_template('index.html',
                           predictions=predictions,
                           actual_vs_pred=actual_vs_pred,
                           labels=labels,
                           counts=counts)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
