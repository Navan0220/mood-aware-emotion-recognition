import os
import librosa
import numpy as np

FEATURE_DIR = "D:/data/processed/features"
RAW_DIR = "D:/data/raw/audio"  # RAVDESS root

os.makedirs(FEATURE_DIR, exist_ok=True)

# Emotion mapping from RAVDESS
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

features, labels = [], []

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        print(f"❌ Error with {file_path}: {e}")
        return None

print("🔍 Starting feature extraction...")

# Walk through all Actor_X folders
for root, dirs, files in os.walk(RAW_DIR):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) > 2:
                emotion_id = parts[2]  # third field = emotion
                emotion_label = EMOTION_MAP.get(emotion_id, "unknown")

                file_path = os.path.join(root, file)
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(emotion_label)
                    print(f"🎵 Processed: {file} -> Label: {emotion_label}")

features = np.array(features)
labels = np.array(labels)

np.save(os.path.join(FEATURE_DIR, "features.npy"), features)
np.save(os.path.join(FEATURE_DIR, "labels.npy"), labels)

print("✅ Feature extraction complete!")
print("   Features shape:", features.shape)
print("   Labels shape:", labels.shape)
print("   Saved to:", FEATURE_DIR)
