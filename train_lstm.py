import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
if np.__version__.startswith("2"):
    print(f"\n[CRITICAL] Incompatible NumPy version detected: {np.__version__}")
    print("TensorFlow requires NumPy 1.x. Please run: pip install \"numpy<2.0\"")
    sys.exit(1)

import tensorflow as tf

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
except ImportError:
    print("\n[ERROR] scikit-learn not found. Please run: pip install scikit-learn")
    sys.exit(1)

try:
    from tensorflow import keras
except ImportError:
    import keras

try:
    import h5py
except ImportError as e:
    print(f"\n[ERROR] h5py failed to import: {e}")
    if "DLL" in str(e):
        print("FIX: Incompatibility detected. Run: pip install \"h5py==3.10.0\" \"numpy<2.0\"")
    else:
        print("Please run: pip install h5py")
    sys.exit(1)

# Define constants matching Drowsiness_Detection.py
SEQUENCE_LEN = 30
FEATURE_COUNT = 10 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "fatigue_data.csv")
# Features: [ear, slope, ear_std, perclos, mar, tilt_angle, blink_rate, ml_drowsy, heart_rate, hrv]

def create_lstm_model():
    model = keras.models.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LEN, FEATURE_COUNT)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid') # Output: Probability of fatigue (0.0 to 1.0)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
    return model

if __name__ == "__main__":
    print("Creating LSTM model architecture...")
    model = create_lstm_model()
    model.summary()
    
    if os.path.exists(CSV_FILE):
        print(f"Loading real data from '{CSV_FILE}'...")
        try:
            # Assumes CSV format: 10 feature columns, last column is label (0 or 1)
            data = np.loadtxt(CSV_FILE, delimiter=",", skiprows=1)
            
            # Reshape for LSTM: (Samples, Sequence_Len, Features)
            # This requires the CSV to be organized in blocks of 30 rows per sample
            num_samples = len(data) // SEQUENCE_LEN
            data = data[:num_samples * SEQUENCE_LEN] # Trim excess

            X = data[:, :-1].reshape(num_samples, SEQUENCE_LEN, FEATURE_COUNT)
            # Labels: usually we take the label of the last frame in the sequence
            y = data[:, -1].reshape(num_samples, SEQUENCE_LEN)[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            print(f"Loaded {num_samples} sequences. Split into {len(X_train)} training and {len(X_test)} testing samples.")
        except Exception as e:
            print(f"Error reading CSV: {e}. Falling back to synthetic.")
            X_train = np.random.rand(100, SEQUENCE_LEN, FEATURE_COUNT).astype(np.float32)
            y_train = np.random.rand(100).astype(np.float32)
    else:
        print(f"'{CSV_FILE}' not found. Generating synthetic data...")
        X_train = np.random.rand(100, SEQUENCE_LEN, FEATURE_COUNT).astype(np.float32) # type: ignore
        y_train = np.random.rand(100).astype(np.float32) # type: ignore

    print("Starting training...")
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

    # Evaluation & Metrics (Requirement 9)
    print("\n--- Model Evaluation ---")
    if 'X_test' in locals() and 'y_test' in locals():
        print("Evaluating on test data...")
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_true = (y_test > 0.5).astype(int)
        print(classification_report(y_true, y_pred, target_names=['Alert', 'Fatigued']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    else:
        print("Evaluating on training data (no test set available)...")
        y_pred_prob = model.predict(X_train)
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_true = (y_train > 0.5).astype(int)
        print(classification_report(y_true, y_pred, target_names=['Alert', 'Fatigued']))

    model.save(os.path.join(BASE_DIR, "fatigue_lstm.keras"))
    print("SUCCESS: Saved 'fatigue_lstm.keras'")