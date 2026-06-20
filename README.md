# Voyage Analytics: Integrating MLOps in Travel

**Productionization of ML Systems** — a capstone project applying end-to-end MLOps practices (model training, experiment tracking, API serving, containerization, orchestration, and CI/CD) to a travel booking dataset.

## Project Overview

This project uses three datasets — `users.csv`, `flights.csv`, and `hotels.csv` — to build and deploy three machine learning models, each wrapped with proper MLOps tooling:

1. **Flight Price Regression** — predicts flight prices based on route, flight type, agency, and distance
2. **Gender Classification** — predicts a user's gender from demographic/name-derived features, used to fill in unlabeled records
3. **Hotel Recommendation** — a content-based recommender that matches users to hotels based on historical price and trip-length preferences

Each model is tracked with MLflow, the flight price model is served via a Flask REST API, containerized with Docker, deployable via Kubernetes, and the training pipeline is automated with Apache Airflow. CI/CD is handled with GitHub Actions, and the recommendation model is exposed through an interactive Streamlit app.

## Tech Stack

| Purpose | Tool |
|---|---|
| Modeling | scikit-learn (RandomForest, LogisticRegression, LinearRegression) |
| Experiment Tracking | MLflow |
| API Serving | Flask |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Pipeline Automation | Apache Airflow |
| CI/CD | GitHub Actions |
| Interactive App | Streamlit |

## Project Structure

```
voyage-analytics-mlops/
├── data/                   # Raw and processed datasets (gitignored CSVs)
├── mlflow/                 # Experiment tracking (gitignored)
├── notebooks/              # EDA scripts
│   ├── 01_eda_flights.py
│   ├── 02_eda_users.py
│   └── 03_eda_recommendation.py
├── src/                    # Model training scripts
│   ├── train_regression.py
│   ├── train_classification.py
│   └── train_recommendation.py
├── api/                    # Flask REST API
│   ├── app.py
│   ├── test_app.py
│   ├── requirements.txt
│   └── *.pkl                # Saved trained models (generated, gitignored)
├── docker/
│   └── Dockerfile
├── k8s/                     # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
├── airflow/
│   ├── dags/
│   │   └── flight_price_pipeline.py
│   ├── docker-compose.yaml
│   └── Dockerfile
├── streamlit_app/
│   └── app.py
├── .github/workflows/
│   └── ci.yml
├── requirements.txt
└── README.md
```

## Datasets

| File | Rows | Description |
|---|---|---|
| `users.csv` | 1,340 | User demographics: company, name, gender, age |
| `flights.csv` | 271,888 | Flight bookings: route, type, price, distance, agency, date |
| `hotels.csv` | 40,552 | Hotel stays: hotel name, city, days, price, date |

**Note:** Each city in this dataset maps to exactly one hotel (9 cities, 9 hotels). This shaped the recommendation approach — see [Model Notes](#model-notes) below.

## Setup

```bash
git clone https://github.com/<your-username>/voyage-analytics-mlops.git
cd voyage-analytics-mlops

python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Place `users.csv`, `flights.csv`, and `hotels.csv` in the `data/` folder.

## Running the Models

```bash
python src/train_regression.py        # Flight price regression
python src/train_classification.py    # Gender classification
python src/train_recommendation.py    # Hotel recommendation
```

View experiment tracking:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://127.0.0.1:5000`.

## Running the API

```bash
cd api
python app.py
```

Test it:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"economic","agency":"Rainbow","distance":676.53,"month":9,"day_of_week":3}'
```

## Running with Docker

```bash
docker build -t flight-price-api -f docker/Dockerfile .
docker run -p 5000:5000 flight-price-api
```

## Running on Kubernetes (Minikube)

```bash
minikube start --driver=docker
minikube image load flight-price-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
minikube service flight-price-api-service --url
```

## Running the Airflow Pipeline

```bash
cd airflow
docker compose up airflow-init
docker compose up -d
```
Open `http://localhost:8080` (login: `airflow` / `airflow`), enable and trigger the `flight_price_training_pipeline` DAG.

## Running the Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```
Open `http://localhost:8501`.

## CI/CD

Every push and pull request to `main` triggers `.github/workflows/ci.yml`, which:
- Installs API dependencies
- Lints with flake8
- Runs unit tests (`api/test_app.py`)
- Verifies the Docker image builds successfully

## Model Notes

- **Flight price regression**: RandomForest achieved near-perfect fit (R² ≈ 1.0), since price appears largely deterministic from `flightType`, `agency`, and `distance` in this dataset. Linear Regression (R² ≈ 0.92) confirms a mostly linear relationship.
- **Gender classification**: ~33% of users have unlabeled (`"none"`) gender. The model is trained on labeled users and used to predict the rest. A baseline using only `company`/`age` performs near chance (~50%), since neither has a real relationship to gender; adding a name-derived feature (last letter of first name) improved accuracy to ~74%. This reflects a known but non-generalizable heuristic (works best for Western/English naming conventions) — noted as a limitation.
- **Hotel recommendation**: Since each city has exactly one hotel, traditional collaborative filtering doesn't apply meaningfully. Instead, a content-based approach matches each user's historical average price paid and trip length against each hotel's typical price/stay profile using scaled Euclidean distance. The Streamlit app also supports a manual-input mode for new users with no booking history (cold-start case).

