import joblib
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model/xgb_model.pkl")

# =========================
# FEATURE NAMES (IMPORTANT)
# =========================
features = [
    "PRG", "PL", "PR", "SK", "TS", "M11", "BD2", "Age",
    "Glucose_BMI", "Age_BMI", "Glucose_Age"
]

# =========================
# HANDLE CALIBRATED MODEL
# =========================
# If wrapped inside CalibratedClassifierCV
if hasattr(model, "base_estimator"):
    model = model.base_estimator
elif hasattr(model, "estimator"):
    model = model.estimator

# =========================
# GET IMPORTANCE
# =========================
importance = model.feature_importances_

# Create DataFrame
df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=True)

# =========================
# PLOT
# =========================
plt.figure(figsize=(8, 6))

plt.barh(df["Feature"], df["Importance"])

plt.xlabel("Importance (Gain)")
plt.title("XGBoost Feature Importance")

plt.tight_layout()

# =========================
# SAVE FIGURE
# =========================
plt.savefig("feature_importance.png", dpi=300)
plt.show()