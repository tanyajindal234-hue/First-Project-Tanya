import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app

@pytest.fixture
def client():
    app = create_app(dataset_dependency=lambda: None)
    return TestClient(app)

def test_health_check(client):
    """Test the Phase 6 health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}

def test_ai_recommendation_endpoint_exists(client):
    """
    Test that the AI recommendation endpoint exists and handles empty datasets.
    This verifies the integration between Phase 3 and Phase 4 (indirectly).
    """
    # Using an empty dataset via dependency override in fixture
    payload = {
        "place": "Bangalore",
        "rating": 4.0,
        "cuisines": ["Indian"]
    }
    response = client.post("/api/v1/recommendations/ai", json=payload)
    # Should return empty recommendations when dataset is empty/None
    assert response.status_code == 200
    assert "recommendations" in response.json()
