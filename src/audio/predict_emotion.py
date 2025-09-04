import os
import numpy as np
import librosa
import joblib

# Paths
MODEL_PATH = "D:/data/models/emotion_model.pkl"

# Load trained model
print("🔍 Loading model...")
clf = joblib.load(MODEL_PATH)

# Function to extract MFCC features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

# Predict emotion
def predict_emotion(file_path):
    features = extract_features(file_path)
    prediction = clf.predict(features)[0]
    return prediction

if __name__ == "__main__":
    # Example: test with a recorded file
    test_file = r"D:\data\raw\audio\audio_speech_actors_01-24\Actor_01\03-01-05-01-01-01-01.wav"

  # change this to your recorded file
    print(f"🎵 Testing on: {test_file}")
    emotion = predict_emotion(test_file)
    print(f"😃 Predicted Emotion: {emotion}")
