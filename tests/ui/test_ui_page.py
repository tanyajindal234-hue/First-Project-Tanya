from fastapi.testclient import TestClient

from src.api.app import create_app


def test_ui_page_renders_root_and_references_endpoints():
    app = create_app(dataset_dependency=lambda: None)  # dataset not used for GET /
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Title text for the React-powered UI
    assert "Zomato AI Recommendation" in html

    # Root div for React
    assert 'id="root"' in html

    # JS should reference the backend endpoints
    assert "/api/v1/recommendations" in html
    assert "/api/v1/places" in html

