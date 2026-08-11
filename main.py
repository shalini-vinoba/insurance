import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load Data
df = pd.read_csv("life_insurance.csv")

print("Original Data Loaded:", df.shape)
print("Columns:", df.columns.tolist())

# Data Cleaning
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

print("After Cleaning:", df.shape)

# Target column
target = "actual_lapse_occurred"

# Remove ID / timestamp columns
drop_columns = ["policy_id", "decision_timestamp"]

for col in drop_columns:
    if col in df.columns:
        df = df.drop(col, axis=1)

# Convert categorical columns to numbers
for c in df.select_dtypes(include=["object", "string"]):
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

print("After Encoding:")
print(df.head())

# Feature Selection
corr = df.corr(numeric_only=True)[target].abs()

features = corr[corr > 0.1].index.drop(target)

print("\nCorrelations:")
print(corr)

print("\nSelected Features:")
print(list(features))

# Split Data
x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(solver="liblinear", max_iter=1000)
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))

# Save Model
joblib.dump(model, "insurance_model.pkl")

print("\ninsurance_model.pkl file created successfully!")

