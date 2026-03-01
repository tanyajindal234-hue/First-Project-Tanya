import math

import pandas as pd

from src.data_access.phase1_preprocessing import clean_restaurant_dataframe


def test_clean_restaurant_dataframe_basic_rules():
    # Arrange: create a small raw-like dataframe
    df = pd.DataFrame(
        {
            "name": ["Spicy House", "Spicy House", "Sweet Corner"],
            "location": ["Indiranagar", "Indiranagar", "Koramangala"],
            "rate": ["4.1/5", "4.1/5", "NEW"],
            "cuisines": [
                "North Indian, Chinese",
                "North Indian, Chinese",
                "Desserts , Bakery",
            ],
            "approx_cost(for two people)": ["1,500", "1,500", "800"],
        }
    )

    # Act
    cleaned = clean_restaurant_dataframe(df)

    # Assert: duplicates removed (two identical Spicy House rows -> one)
    assert len(cleaned) == 2
    assert sorted(cleaned["name"].unique().tolist()) == ["Spicy House", "Sweet Corner"]

    # Assert: price is numeric without thousand separators
    spicy_row = cleaned[cleaned["name"] == "Spicy House"].iloc[0]
    sweet_row = cleaned[cleaned["name"] == "Sweet Corner"].iloc[0]

    assert spicy_row["price"] == 1500.0
    assert sweet_row["price"] == 800.0

    # Assert: rating is stored correctly (4.1 not "4.1/5")
    assert math.isclose(spicy_row["rating"], 4.1)
    # "NEW" rating becomes None/NaN
    assert math.isnan(sweet_row["rating"])

    # Assert: cuisines are split and cleaned
    assert spicy_row["cuisines_clean"] == ["North Indian", "Chinese"]
    assert sweet_row["cuisines_clean"] == ["Desserts", "Bakery"]

