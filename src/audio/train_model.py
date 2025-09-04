import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Paths
FEATURES_PATH = r"D:\data\processed\features"
MODEL_PATH = r"D:\data\models\emotion_model.pkl"

# Load combined features and labels
features_file = os.path.join(FEATURES_PATH, "features.npy")
labels_file = os.path.join(FEATURES_PATH, "labels.npy")

print("🔍 Loading data...")
X = np.load(features_file)
y = np.load(labels_file)

print(f"✅ Features shape: {X.shape}, Labels shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
print("🚀 Training model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"💾 Model saved to: {MODEL_PATH}")
