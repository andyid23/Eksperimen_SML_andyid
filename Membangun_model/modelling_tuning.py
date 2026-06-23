import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ====== SETUP MLFLOW ======
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Kriteria2_RF_Tuning")

# ====== LOAD DATA ======
df = pd.read_csv("namadataset_preprocessing/processed_data.csv")
X = df.drop(columns=["Post_Semester_GPA"])
y = df["Post_Semester_GPA"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== HYPERPARAMETER TUNING ======
param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# ====== EVALUASI ======
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best params: {grid.best_params_}")
print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# ====== LOG KE MLFLOW ======
with mlflow.start_run(run_name="RandomForest_Tuned_ManualLogging"):
    # Log hyperparameters
    for k, v in grid.best_params_.items():
        mlflow.log_param(k, v)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    # Log dataset
    mlflow.log_artifact("namadataset_preprocessing/processed_data.csv")

    # === FEATURE IMPORTANCE PLOT ===
    importances = best_model.feature_importances_
    feats = X.columns
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feats, importances, color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    # === ACTUAL vs PREDICTED PLOT ===
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual Post_Semester_GPA")
    ax.set_ylabel("Predicted Post_Semester_GPA")
    ax.set_title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    mlflow.log_artifact("actual_vs_predicted.png")
    plt.close()

    print(f"✅ Run selesai — Buka UI: http://127.0.0.1:5000")
