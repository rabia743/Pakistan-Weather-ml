import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# accuracy_score, confusion_matrix, classification_report ya model evualtion ha
BASE_DIR = Path(__file__).resolve().parent

weather = pd.read_csv(BASE_DIR / "weather_feature_engineered.csv")
# Features
# Independent variables
features = ["Tavg", "Prcp", "Wspd", "Humidity", "Cloud_Cover", "Tmin", "Tmax", "Dew_Point"]
X = weather[features]
# Derive the target using the same play-condition rules as weather.py so
# training does not reuse stale labels from an older feature-engineered file.
y = (
    (weather["Tavg"] >= 15) &
    (weather["Tavg"] <= 35) &
    (weather["Tmax"] <= 35) &
    (weather["Prcp"] <= 5) &
    (weather["Wspd"] <= 30) &
    (weather["Humidity"] <= 85) &
    (weather["Cloud_Cover"] <= 90)
).astype(int)
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# KNN Model
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, metric="euclidean")
)
# Training
model.fit(X_train, y_train)
# Test Data Prediction
y_pred = model.predict(X_test)
# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Save the trained model for the Flask API.
model_path = BASE_DIR / "weather_knn_model.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved as {model_path}")

print("\nEnter values for a new prediction.")
user_values = {}
for feature in features:
    minimum = float(weather[feature].min())
    maximum = float(weather[feature].max())
    user_values[feature] = float(input(f"{feature} ({minimum:g} to {maximum:g}): "))

new_weather = pd.DataFrame([user_values], columns=features)
prediction = model.predict(new_weather)[0]
print("Football Play:", "Yes" if prediction == 1 else "No")
