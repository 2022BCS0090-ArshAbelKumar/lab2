import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

df = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

joblib.dump(model, "outputs/model/model.pkl")

results = {
    "mse": mse,
    "r2": r2
}

with open("outputs/results/results.json", "w") as f:
    json.dump(results, f, indent=4)
