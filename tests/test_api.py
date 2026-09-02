from fastapi.testclient import TestClient
import api.main as main


def test_health_endpoint():
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_risk_bucket():
    assert main._risk_bucket(0.10) == ("LOW", "APPROVE")
    assert main._risk_bucket(0.50) == ("MEDIUM", "REVIEW")
    assert main._risk_bucket(0.90) == ("HIGH", "BLOCK")
