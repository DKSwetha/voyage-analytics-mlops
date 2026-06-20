import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---- Load data ----
df = pd.read_csv("data/users.csv")

# "none" = unlabeled rows we'll predict later, not a training class
labeled_df = df[df["gender"] != "none"].copy()
unlabeled_df = df[df["gender"] == "none"].copy()

print(f"Training on {len(labeled_df)} labeled rows")
print(f"Will predict gender for {len(unlabeled_df)} unlabeled rows")

# ---- Name-derived feature (used only in the "enhanced" version) ----
def add_name_features(data):
    data = data.copy()
    data["first_name"] = data["name"].str.split().str[0]
    data["name_last_letter"] = data["first_name"].str[-1].str.lower()
    data["name_length"] = data["first_name"].str.len()
    return data

labeled_df = add_name_features(labeled_df)
unlabeled_df = add_name_features(unlabeled_df)

target = "gender"
le = LabelEncoder()
y = le.fit_transform(labeled_df[target])  # male/female -> 0/1

# Two feature sets to compare
feature_sets = {
    "baseline_company_age": {
        "categorical": ["company"],
        "numeric": ["age"]
    },
    "enhanced_with_name": {
        "categorical": ["company", "name_last_letter"],
        "numeric": ["age", "name_length"]
    }
}

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

mlflow.set_experiment("gender-classification")

best_model = None
best_f1 = -1
best_combo_name = None

for fs_name, fs in feature_sets.items():
    X = labeled_df[fs["categorical"] + fs["numeric"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), fs["categorical"]),
        ],
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for model_name, model in models.items():
        run_name = f"{fs_name}_{model_name}"
        with mlflow.start_run(run_name=run_name):
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            print(f"\n{run_name}: Accuracy={acc:.4f}, F1={f1:.4f}")
            print(classification_report(y_test, preds, target_names=le.classes_))

            mlflow.log_param("feature_set", fs_name)
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(pipeline, "model")

            if f1 > best_f1:
                best_f1 = f1
                best_model = pipeline
                best_combo_name = run_name

print(f"\nBest combo: {best_combo_name} (F1={best_f1:.4f})")

# ---- Save best model + label encoder ----
joblib.dump(best_model, "api/gender_model.pkl")
joblib.dump(le, "api/gender_label_encoder.pkl")

# ---- Predict the unlabeled ("none") rows using the best model ----
X_unlabeled = unlabeled_df[
    feature_sets["baseline_company_age" if "baseline" in best_combo_name else "enhanced_with_name"]["categorical"] +
    feature_sets["baseline_company_age" if "baseline" in best_combo_name else "enhanced_with_name"]["numeric"]
]
predicted_labels = le.inverse_transform(best_model.predict(X_unlabeled))
unlabeled_df["predicted_gender"] = predicted_labels
unlabeled_df.to_csv("data/users_gender_predicted.csv", index=False)
print(f"\nPredictions for {len(unlabeled_df)} unlabeled users saved to data/users_gender_predicted.csv")