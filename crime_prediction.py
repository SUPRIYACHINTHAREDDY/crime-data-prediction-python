import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sample Data (replace with actual dataset)
data = pd.read_csv("crime_data.csv")  # Columns: ['Area', 'Crime_Type', 'Time', 'Severity']

# Encode categorical columns
data = pd.get_dummies(data, drop_first=True)

X = data.drop("Severity", axis=1)
y = data["Severity"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
