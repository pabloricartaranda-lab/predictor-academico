import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# =========================
# Cargar datasets
# =========================
df_train = pd.read_csv(BASE_DIR / "dataset_train.csv")
df_test = pd.read_csv(BASE_DIR / "dataset_test_adaptado.csv")

# =========================
# Entrenar modelo
# =========================
model = smf.ols(
    "GPA ~ Study_Hours_Per_Day + "
    "Sleep_Hours_Per_Day + "
    "Extracurricular_Hours_Per_Day + "
    "Social_Hours_Per_Day + "
    "Stress_Level",
    data=df_train
).fit()

# =========================
# Predicciones
# =========================
y_train = df_train["GPA"]
y_test = df_test["GPA"]

y_pred_train = model.predict(df_train)
y_pred_test = model.predict(df_test)

# =========================
# Métricas
# =========================
r2_train = r2_score(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# =========================
# Resultados
# =========================
print(model.summary())

print("\n" + "="*50)
print("               MÉTRICAS TRAIN")
print("="*50)
print(f"R²   : {r2_train:.4f}")
print(f"MAE  : {mae_train:.4f}")
print(f"RMSE : {rmse_train:.4f}")

print("\n" + "="*50)
print("                MÉTRICAS TEST")
print("="*50)
print(f"R²   : {r2_test:.4f}")
print(f"MAE  : {mae_test:.4f}")
print(f"RMSE : {rmse_test:.4f}")

# =========================
# Guardar modelo
# =========================
joblib.dump(model, BASE_DIR / "modelo.pkl")

print("\nModelo guardado como modelo.pkl")