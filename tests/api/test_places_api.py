import pandas as pd
from fastapi.testclient import TestClient

from src.api.app import create_app


def test_list_places_returns_unique_sorted_locations():
    df = pd.DataFrame(
        {
            "location": [
                "Indiranagar",
                "Koramangala",
                "Indiranagar",
                "Whitefield ",
                None,
                "",
            ]
        }
    )

    app = create_app(dataset_dependency=lambda: df)
    client = TestClient(app)

    resp = client.get("/api/v1/places")
    assert resp.status_code == 200

    places = resp.json()
    # Should be unique, trimmed, and sorted
    assert places == ["Indiranagar", "Koramangala", "Whitefield"]

