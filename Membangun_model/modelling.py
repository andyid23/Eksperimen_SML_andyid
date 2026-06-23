import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ====== SETUP MLFLOW ======
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Kriteria2_RF_Basic")

# ====== LOAD DATA ======
df = pd.read_csv("namadataset_preprocessing/processed_data.csv")
X = df.drop(columns=["Post_Semester_GPA"])
y = df["Post_Semester_GPA"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== TRAIN + LOG (AUTOLOG) ======
with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
    mlflow.sklearn.autolog()

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Manual logging juga (redundan tapi aman)
    mlflow.log_metric("manual_mse", mse)
    mlflow.log_metric("manual_r2", r2)

    print(f"✅ Run selesai — MSE: {mse:.4f}, R2: {r2:.4f}")
    print(f"📌 Buka UI: http://127.0.0.1:5000")
