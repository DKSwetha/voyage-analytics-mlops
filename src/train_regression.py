import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ---- Load data ----
df = pd.read_csv("data/flights.csv")

# ---- Feature engineering ----
# Extract date features
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

# Drop redundant column (time and distance are ~99.999% correlated, keep distance)
features = ["from", "to", "flightType", "agency", "distance", "month", "day_of_week"]
target = "price"

X = df[features]
y = df[target]

categorical_cols = ["from", "to", "flightType", "agency"]
numeric_cols = ["distance", "month", "day_of_week"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough"  # keeps numeric_cols as-is
)

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Models to compare ----
models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

mlflow.set_experiment("flight-price-prediction")

best_model = None
best_rmse = float("inf")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(pipeline, "model")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_model_name = name

# ---- Save best model for the Flask API ----
joblib.dump(best_model, "api/model.pkl")
print(f"\nBest model: {best_model_name} (RMSE={best_rmse:.2f}) saved to api/model.pkl")