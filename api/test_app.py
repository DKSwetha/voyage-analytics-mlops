import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_valid_input(client):
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "economic",
        "agency": "Rainbow",
        "distance": 676.53,
        "month": 9,
        "day_of_week": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)


def test_predict_missing_fields(client):
    response = client.post("/predict", json={"from": "Recife (PE)"})
    assert response.status_code == 400
    assert "error" in response.get_json()
