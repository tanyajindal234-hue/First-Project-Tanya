import pandas as pd
from fastapi.testclient import TestClient

from src.api.app import create_app


def build_synthetic_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [
                "Spicy House",
                "Spicy House",  # duplicate row
                "Okay House",
                "Far Away Diner",
            ],
            "location": [
                "Indiranagar",
                "Indiranagar",
                "Indiranagar",
                "Koramangala",
            ],
            "rating": [4.5, 4.5, 4.0, 4.6],
            "price": [1500.0, 1500.0, 1400.0, 900.0],
            "cuisines_clean": [
                ["North Indian", "Chinese"],
                ["North Indian", "Chinese"],
                ["Chinese"],
                ["North Indian"],
            ],
        }
    )


def test_recommendations_endpoint_basic_flow():
    dataset = build_synthetic_dataset()

    # Override dataset dependency using the application factory
    app = create_app(dataset_dependency=lambda: dataset)
    client = TestClient(app)

    payload = {
        "place": "Indiranagar",
        "rating": 4.0,
        "min_price": 200.0,
        "max_price": 2000.0,
        "cuisines": ["North Indian"],
    }

    resp = client.post("/api/v1/recommendations", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert "recommendations" in body
    recs = body["recommendations"]

    # At least one recommendation should be returned
    assert len(recs) >= 1

    # All recommendations should respect location and rating constraints
    for r in recs:
        assert "Indiranagar" in (r["location"] or "")
        assert r["rating"] is None or r["rating"] >= 4.0

    # Duplicate restaurant names should not appear more than once
    names = [r["name"] for r in recs]
    assert names.count("Spicy House") == 1

