import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import joblib  # for saving scaler and encoder

# Load dataset
df = pd.read_csv('emotions.csv')  # Make sure this file is present

# Feature selection
X = df.loc[:, 'fft_0_b':'fft_749_b'].values
y = df['label'].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and encoder for inference
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.1, random_state=42, stratify=y_cat)

# Reshape for GRU
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the GRU model
model = Sequential([
    Bidirectional(GRU(512, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)), input_shape=(1, X_train.shape[2])),
    Dropout(0.5),
    Bidirectional(GRU(256, kernel_regularizer=regularizers.l2(0.01))),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
])

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred_labels)
print("Accuracy:", round(acc * 100, 2), "%")
print(classification_report(y_true, y_pred_labels, target_names=encoder.classes_))

# Save model
model.save('gru_emotion_model_v3.h5')
print("Model, scaler, and encoder saved.")
