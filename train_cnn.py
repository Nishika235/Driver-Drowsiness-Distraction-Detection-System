import os
import sys

# TensorFlow environment settings (safe to keep)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
if np.__version__.startswith("2"):
    print(f"\n[CRITICAL] Incompatible NumPy version detected: {np.__version__}")
    print("TensorFlow requires NumPy 1.x. Please run: pip install \"numpy<2.0\"")
    sys.exit(1)

import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

try:
    import h5py
except ImportError as e:
    print(f"\n[ERROR] h5py failed to import: {e}")
    if "DLL" in str(e):
        print("FIX: Incompatibility detected. Run: pip install \"h5py==3.10.0\" \"numpy<2.0\"")
    else:
        print("Please run: pip install h5py")
    sys.exit(1)

# -------------------- Constants --------------------
IMG_SIZE = 24
CHANNELS = 1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "drowsiness_model.keras")
DATA_DIR = os.path.join(BASE_DIR, "dataset")   # dataset/Open, dataset/Closed
CACHE_FILE = os.path.join(BASE_DIR, "dataset_cache.npz")

# -------------------- Data Loading --------------------
def load_real_data():
    data = []
    labels = []
    classes = ["Closed", "Open"]  # 0: Closed, 1: Open

    print(f"Loading images from '{DATA_DIR}'...")
    for category in classes:
        path = os.path.join(DATA_DIR, category)
        class_num = classes.index(category)

        if not os.path.exists(path):
            print(f"  [WARN] Folder '{path}' not found. Skipping.")
            continue

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(class_num)
            except:
                pass

    if len(data) == 0:
        return None, None

    data = np.array(data, dtype=np.float32)
    data = data.reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)
    data /= 255.0  # normalize
    labels = np.array(labels)

    return data, labels

def get_data():
    return load_real_data()

# -------------------- Main --------------------
if __name__ == "__main__":

    print("Building CNN model architecture...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',
               input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # -------------------- Dataset Handling --------------------
    if os.path.exists(DATA_DIR):
        if os.path.exists(CACHE_FILE):
            print(f"Loading data from cache: {CACHE_FILE}")
            cache = np.load(CACHE_FILE)
            X, y = cache['X'], cache['y']
        else:
            print("Cache not found. Loading images...")
            X, y = get_data()
            if X is not None:
                print("Saving cache...")
                np.savez_compressed(CACHE_FILE, X=X, y=y)

        if X is not None and y is not None:
            print(f"Loaded {len(X)} images.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            print("No real images found. Using synthetic data.")
            X_train = np.random.rand(100, IMG_SIZE, IMG_SIZE, CHANNELS).astype(np.float32)
            y_train = np.random.randint(2, size=100).astype(np.float32)
    else:
        print(f"'{DATA_DIR}' folder not found. Using synthetic data.")
        X_train = np.random.rand(100, IMG_SIZE, IMG_SIZE, CHANNELS).astype(np.float32)
        y_train = np.random.randint(2, size=100).astype(np.float32)

    # -------------------- Training --------------------
    print("Starting training...")
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

    # -------------------- Evaluation --------------------
    print("\n--- Model Evaluation ---")
    if 'X_test' in locals() and 'y_test' in locals():
        print("Evaluating on test data...")
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        print(classification_report(y_test, y_pred, target_names=["Closed", "Open"]))
    else:
        print("Evaluating on training data (no test set available)...")
        y_pred = (model.predict(X_train) > 0.5).astype(int)
        print(classification_report(y_train, y_pred, target_names=["Closed", "Open"]))

    # -------------------- Save Model --------------------
    model.save(MODEL_FILE)

    print(f"\nSUCCESS: Model saved as '{MODEL_FILE}'")
    print("You can now run app.py for real-time detection.")
