import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load dataset
data = pd.read_csv("transactions.csv")

print(data.head())
print(data.info())
print(data["fraud"].value_counts())


# Encode text columns
encoders = {}

categorical_columns = [
    "transaction_type",
    "merchant_category",
    "location",
    "card_type",
    "is_foreign_transaction"
]

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le


# Split input and output
X = data.drop("fraud", axis=1)
y = data["fraud"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# Test model
y_pred = model.predict(X_test)

print("Accuracy Score =", accuracy_score(y_test, y_pred))
print("Precision Score =", precision_score(y_test, y_pred))
print("Recall Score =", recall_score(y_test, y_pred))
print("F1 Score =", f1_score(y_test, y_pred))


# Save model and encoders
joblib.dump(model, "transaction_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model trained and saved successfully!")