#!/usr/bin/env python3
"""extract_features.py

Script to extract 13‑dimensional MFCC feature vectors from the `dataset/evet`
and `dataset/hayir` folders.

- Loads each `.wav` file at 16 kHz using librosa.
- Computes MFCCs (`n_mfcc=13`).
- Averages over the time axis to obtain a single 13‑element vector per file.
- Assigns label 1 to "evet" (yes) samples and 0 to "hayir" (no) samples.
- Saves the feature matrix X and label vector y as `X.npy` and `y.npy`.
- Prints the number of processed "evet" and "hayir" files.
"""

import os
import numpy as np
import librosa

# Configuration
SR = 16000            # Target sampling rate
N_MFCC = 13           # Number of MFCC coefficients
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")
YES_DIR = os.path.join(DATASET_ROOT, "evet")
NO_DIR = os.path.join(DATASET_ROOT, "hayir")

def collect_files(directory: str):
    """Return a sorted list of absolute paths to `.wav` files in *directory*."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith('.wav')
    )

def extract_vector(file_path: str) -> np.ndarray:
    """Load *file_path*, compute MFCCs, and return the mean vector (shape (13,))."""
    y, sr = librosa.load(file_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

def main():
    yes_files = collect_files(YES_DIR)
    no_files = collect_files(NO_DIR)
    X_list = []
    y_list = []
    for fp in yes_files:
        X_list.append(extract_vector(fp))
        y_list.append(1)
    for fp in no_files:
        X_list.append(extract_vector(fp))
        y_list.append(0)
    X = np.stack(X_list)
    y = np.array(y_list)
    np.save(os.path.join(DATASET_ROOT, "X.npy"), X)
    np.save(os.path.join(DATASET_ROOT, "y.npy"), y)
    print(f"Processed {len(yes_files)} 'evet' files and {len(no_files)} 'hayir' files.")
    print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

if __name__ == "__main__":
    main()
