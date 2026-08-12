# train_model.py
"""Train a machine learning model for voice commands.

Loads feature matrix X and labels y from the `dataset` directory, scales the features,
trains a RandomForestClassifier (can be swapped for an SVC), evaluates on the training
set, prints accuracy and confusion matrix, and saves the trained model and scaler.
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # Paths
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    X_path = os.path.join(dataset_dir, "X.npy")
    y_path = os.path.join(dataset_dir, "y.npy")

    # Load data
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # Stratified train / test split (20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale features (fit on training set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model – SVC
    model = SVC(kernel='rbf', C=1.0, probability=True,
                class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate on test data
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["HAYIR", "EVET"], digits=4)
    print(f"Test accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    # Save the trained model and scaler
    model_path = os.path.join(os.path.dirname(__file__), "voice_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    main()
