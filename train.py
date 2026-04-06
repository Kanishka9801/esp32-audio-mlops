import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import mlflow

# 1. SETUP & PATHS
DATA_PATH = "dataset"
mlflow.set_experiment("ESP32_Audio_Threat_Detection")

# --- FUNCTIONS ---
def augment_audio(y, sr):
    y_stretch = librosa.effects.time_stretch(y, rate=0.9)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    noise = y + 0.005 * np.random.randn(len(y))
    return [y, y_stretch, y_pitch, noise]

def extract_logmel(y, sr, n_mels=64):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)
    return np.mean(logmel, axis=1)

with mlflow.start_run():
    print("Starting MLOps Pipeline...")
    
    # 2. DATA LOADING & EXTRACTION
    X, y = [], []
    labels = sorted(os.listdir(DATA_PATH))
    print("Classes found:", labels)

    for label in labels:
        folder = os.path.join(DATA_PATH, label)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            # Suppress warnings for clean pipeline logs
            import warnings
            warnings.filterwarnings("ignore")
            
            signal, sr = librosa.load(file_path, sr=16000)
            augmented_signals = augment_audio(signal, sr)

            for aug_signal in augmented_signals:
                features = extract_logmel(aug_signal, sr)
                X.append(features)
                y.append(label)

    X = np.array(X)
    y_array = np.array(y)
    print("Feature matrix shape:", X.shape)

    # 3. ENCODING & SCALING
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_array)
    joblib.dump(encoder, "label_encoder.save")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.save")

    # 4. MLFLOW LOGGING (Params)
    learning_rate = 0.0008
    epochs = 200
    batch_size = 16
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("classes", labels)

    # 5. MODEL ARCHITECTURE
    model = Sequential([
        Input(shape=(X_train.shape[1],)),   
        Dense(320, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(encoder.classes_), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. TRAINING
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # 7. EVALUATION & METRICS
    loss, accuracy = model.evaluate(X_test, y_test)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_loss", loss)
    print(f"\nFINAL TEST ACCURACY: {accuracy:.4f}")

    # Generate Confusion Matrix (Saved as image instead of shown)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png") # Save for CI/CD
    
    # 8. CONVERT TO TFLITE
    print("Converting model to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    # 9. LOG EVERYTHING TO MLFLOW
    print("Logging artifacts to MLflow...")
    mlflow.log_artifact("label_encoder.save")
    mlflow.log_artifact("scaler.save")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("model.tflite")

    print("Pipeline Execution Complete!")