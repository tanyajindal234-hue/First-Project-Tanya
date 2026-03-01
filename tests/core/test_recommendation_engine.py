import numpy as np
import pandas as pd

from src.core.recommendation_engine import UserPreference, get_candidate_restaurants


def test_get_candidate_restaurants_filters_and_deduplicates():
    # Arrange: build a small synthetic processed-like dataset
    df = pd.DataFrame(
        {
            "name": [
                "Spicy House",
                "Spicy House",  # duplicate
                "Cheap Bites",
                "Far Away Diner",
                "Okay House",
            ],
            "location": [
                "Indiranagar",
                "Indiranagar",
                "Indiranagar",
                "Koramangala",
                "Indiranagar",
            ],
            "rating": [4.5, 4.5, 3.2, 4.6, 4.0],
            "price": [1500.0, 1500.0, 500.0, 800.0, 1400.0],
            # Use numpy arrays here to mimic how the real dataset may be stored
            "cuisines_clean": [
                np.array(["North Indian", "Chinese"]),
                np.array(["North Indian", "Chinese"]),
                np.array(["North Indian"]),
                np.array(["North Indian"]),
                np.array(["Chinese"]),
            ],
        }
    )

    prefs = UserPreference(
        location="Indiranagar",
        min_rating=4.0,
        cuisines=["North Indian"],
        min_price=200.0,
        max_price=2000.0,
    )

    # Act
    candidates = get_candidate_restaurants(df, prefs, top_n=10)

    # Assert: only restaurants in the requested location and with rating >= 4.0
    assert candidates, "Expected at least one candidate"
    for r in candidates:
        assert "Indiranagar" in (r.location or "")
        assert r.rating is None or r.rating >= 4.0

    # Assert: duplicates removed (only one 'Spicy House' despite duplicate rows)
    spicy_names = [r for r in candidates if r.name == "Spicy House"]
    assert len(spicy_names) == 1

    # Assert: scoring prefers higher-rated and better-matching restaurant
    # 'Spicy House' has higher rating and better cuisine match than 'Okay House',
    # so it should appear before 'Okay House' in the sorted list.
    names_in_order = [r.name for r in candidates]
    if "Okay House" in names_in_order:
        assert names_in_order.index("Spicy House") < names_in_order.index("Okay House")

