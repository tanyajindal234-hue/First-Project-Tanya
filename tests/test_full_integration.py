import os
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.data_access.phase1_preprocessing import clean_restaurant_dataframe
from src.core.recommendation_engine import UserPreference, get_candidate_restaurants
from src.llm.orchestrator import generate_llm_recommendations, LLMRecommendation
from src.api.app import create_app
from fastapi.testclient import TestClient

@pytest.fixture
def sample_df():
    data = {
        "name": ["Resto A", "Resto B", "Resto C"],
        "location": ["Bangalore", "Bangalore", "Delhi"],
        "rating": [4.5, 3.8, 4.2],
        "cuisines": ["Italian, Pizza", "Chinese", "Italian"],
        "price": [1200, 800, 1500]
    }
    return pd.DataFrame(data)

def test_full_system_flow(sample_df):
    """
    Integration test: Preprocessing -> Core Engine -> LLM Orchestrator.
    """
    # 1. Phase 1: Preprocessing
    clean_df = clean_restaurant_dataframe(sample_df)
    assert "cuisines_clean" in clean_df.columns
    assert len(clean_df) == 3

    # 2. Phase 2: Core Engine
    prefs = UserPreference(location="Bangalore", min_rating=4.0, cuisines=["Italian"])
    candidates = get_candidate_restaurants(clean_df, prefs)
    
    assert len(candidates) >= 1
    assert candidates[0].name == "Resto A"

    # 3. Phase 4: LLM Orchestrator (Mocked)
    mock_client = MagicMock()
    mock_client.generate.return_value = '[{"index": 0, "reason": "Matches all criteria"}]'
    
    recommendations = generate_llm_recommendations(prefs, candidates, client=mock_client)
    
    assert len(recommendations) == 1
    assert recommendations[0].name == "Resto A"
    assert recommendations[0].reason == "Matches all criteria"

def test_api_integration_with_data(sample_df):
    """
    Integration test: API -> Core Engine -> AI Endpoint.
    """
    # Create app with injected sample data
    app = create_app(dataset_dependency=lambda: clean_restaurant_dataframe(sample_df))
    client = TestClient(app)

    # Mock the LLM call inside the endpoint if possible, or just test the wiring
    # For simplicity here, we test the non-AI endpoint first
    payload = {
        "place": "Bangalore",
        "rating": 4.0,
        "cuisines": ["Italian"]
    }
    
    # Test Phase 3 endpoint
    resp = client.post("/api/v1/recommendations", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["recommendations"]) > 0
    assert data["recommendations"][0]["name"] == "Resto A"

    # Test Phase 6 Health
    health = client.get("/health")
    assert health.status_code == 200
