import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

# =========================
# LOAD SAVED MODEL (SAFE)
# =========================
calibrated_model = joblib.load("model/xgb_model.pkl")

print("✅ Model loaded successfully")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/Paitients_Files_Train.csv")
df.columns = df.columns.str.strip()

# Drop ID
df = df.drop(columns=["ID"], errors="ignore")

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode target
le = LabelEncoder()
df["Sepssis"] = le.fit_transform(df["Sepssis"])

# =========================
# FEATURE ENGINEERING
# =========================
df["Glucose_BMI"] = df["PL"] * df["M11"]
df["Age_BMI"] = df["Age"] * df["M11"]
df["Glucose_Age"] = df["PL"] * df["Age"]

# Features & target
X = df.drop(["Sepssis", "Insurance"], axis=1)
y = df["Sepssis"]

# SAME SPLIT AS TRAINING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# RAW MODEL (FOR COMPARISON ONLY)
# =========================
raw_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    max_depth=5,
    learning_rate=0.03,
    n_estimators=500,
    subsample=0.9,
    colsample_bytree=0.9
)

raw_model.fit(X_train, y_train)

print("✅ Raw model trained for comparison")

# =========================
# GET PROBABILITIES
# =========================
y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
y_prob_raw = raw_model.predict_proba(X_test)[:, 1]

# =========================
# CALIBRATION DATA
# =========================
frac_pos_raw, mean_pred_raw = calibration_curve(y_test, y_prob_raw, n_bins=10)
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, y_prob_cal, n_bins=10)

# =========================
# PLOT
# =========================
plt.figure(figsize=(6, 6))

# Perfect calibration line
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

# Raw model
plt.plot(mean_pred_raw, frac_pos_raw, 's-', label='Uncalibrated XGBoost')

# Calibrated model
plt.plot(mean_pred_cal, frac_pos_cal, 'o-', label='Calibrated (Isotonic)')

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Actual Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.grid()

# =========================
# SAVE FIGURE
# =========================
plt.savefig("calibration_curve.png", dpi=300)
print("📊 Calibration curve saved as calibration_curve.png")

plt.show()