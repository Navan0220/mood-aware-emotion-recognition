# extract_features.py
import os
import librosa
import numpy as np
import pandas as pd

# ✅ Path to your processed audio folder
AUDIO_DIR = r"D:\data\processed\audio"
# ✅ Where to save extracted features
OUTPUT_CSV = r"D:\data\processed\features.csv"

def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    features = []
    labels = []

    for file_name in os.listdir(AUDIO_DIR):
        if file_name.endswith(".wav"):
            file_path = os.path.join(AUDIO_DIR, file_name)
            print(f"Processing {file_name}...")

            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)

                # Use last word before ".wav" as label (e.g. "happy" from "jenani_happy.wav")
                label = file_name.split("_")[-1].replace(".wav", "")
                labels.append(label)

    # Convert to DataFrame
    df = pd.DataFrame(features)
    df["label"] = labels

    # Save as CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Features saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
