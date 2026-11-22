
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load dataset
data = pd.read_csv("data/cleaned_comments.csv")

# Check required columns
if "cleaned_text" not in data.columns or "toxic" not in data.columns:
    raise ValueError(
        "Dataset must contain cleaned_text and toxic columns."
    )

# Prepare features and labels
X = data["cleaned_text"].fillna("")
y = data["toxic"].astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# Make predictions
predictions = model.predict(X_test_vec)

# Evaluate model
print("BASELINE MODEL RESULTS")
print("----------------------")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions))

# Save trained model
joblib.dump(model, "code/baseline_model.pkl")
joblib.dump(vectorizer, "code/tfidf_vectorizer.pkl")

print("\nModel successfully trained and saved!")
