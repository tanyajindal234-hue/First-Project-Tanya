from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class UserPreference:
    """
    Core preference model used by the non-LLM recommendation engine.
    """

    location: Optional[str] = None
    min_rating: Optional[float] = None
    cuisines: List[str] = field(default_factory=list)
    min_price: Optional[float] = None
    max_price: Optional[float] = None

    def normalized_cuisines(self) -> List[str]:
        return [c.strip() for c in self.cuisines if c and c.strip()]


@dataclass
class Restaurant:
    """
    Core restaurant domain model used by the recommendation engine.
    """

    name: str
    location: Optional[str]
    rating: Optional[float]
    price: Optional[float]
    cuisines: List[str]


def _detect_location_column(df: pd.DataFrame) -> Optional[str]:
    """
    Best-effort detection of the location column in the processed dataset.
    """
    for candidate in ("location", "listed_in(city)", "city", "address"):
        if candidate in df.columns:
            return candidate
    return None


def _filter_by_preferences(df: pd.DataFrame, prefs: UserPreference) -> pd.DataFrame:
    """
    Apply hard filters based on user preferences:
    - Location (contains, case-insensitive)
    - Minimum rating
    - Price range
    - At least one matching cuisine (if preferences specify cuisines)
    """
    filtered = df

    # Location filter
    loc_col = _detect_location_column(filtered)
    if prefs.location and loc_col:
        filtered = filtered[
            filtered[loc_col]
            .astype(str)
            .str.contains(prefs.location, case=False, na=False)
        ]

    # Rating filter
    if prefs.min_rating is not None and "rating" in filtered.columns:
        filtered = filtered[filtered["rating"].fillna(0) >= prefs.min_rating]

    # Price range filter
    if "price" in filtered.columns and (
        prefs.min_price is not None or prefs.max_price is not None
    ):
        if prefs.min_price is not None:
            filtered = filtered[filtered["price"].fillna(0) >= prefs.min_price]
        if prefs.max_price is not None:
            filtered = filtered[filtered["price"].fillna(0) <= prefs.max_price]

    # Cuisine filter (at least one overlapping cuisine)
    cuisines = prefs.normalized_cuisines()
    if cuisines and "cuisines_clean" in filtered.columns:
        cuisine_set = set(cuisines)

        def has_overlap(value) -> bool:
            # Handle list/tuple/set, numpy arrays, and comma-separated strings
            if value is None:
                return False
            if hasattr(value, "tolist"):
                seq = value.tolist()
            elif isinstance(value, (list, tuple, set)):
                seq = list(value)
            elif isinstance(value, str):
                seq = [part.strip() for part in value.split(",") if part.strip()]
            else:
                return False

            return bool(cuisine_set.intersection(seq))

        filtered = filtered[filtered["cuisines_clean"].apply(has_overlap)]

    return filtered


def _compute_score(row: pd.Series, prefs: UserPreference) -> float:
    """
    Compute a simple score for a restaurant row based on:
    - Rating (primary factor)
    - Cuisine overlap
    - Price alignment with preferred range
    """
    score = 0.0

    rating = row.get("rating")
    if isinstance(rating, (int, float)) and not pd.isna(rating):
        score += float(rating) * 2.0

    price = row.get("price")
    if isinstance(price, (int, float)) and not pd.isna(price):
        if prefs.min_price is not None or prefs.max_price is not None:
            in_range = True
            if prefs.min_price is not None and price < prefs.min_price:
                in_range = False
            if prefs.max_price is not None and price > prefs.max_price:
                in_range = False
            if in_range:
                score += 1.0
            else:
                score -= 1.0

    cuisines_pref = set(prefs.normalized_cuisines())
    raw_cuisines = row.get("cuisines_clean")

    # Normalize cuisines for scoring
    if raw_cuisines is None:
        cuisines_row = []
    elif hasattr(raw_cuisines, "tolist"):
        cuisines_row = list(raw_cuisines.tolist())
    elif isinstance(raw_cuisines, (list, tuple, set)):
        cuisines_row = list(raw_cuisines)
    elif isinstance(raw_cuisines, str):
        cuisines_row = [
            part.strip() for part in raw_cuisines.split(",") if part.strip()
        ]
    else:
        cuisines_row = []

    if cuisines_pref and cuisines_row:
        matches = len(cuisines_pref.intersection(cuisines_row))
        score += matches * 1.5

    return score


def get_candidate_restaurants(
    df: pd.DataFrame,
    preferences: UserPreference,
    top_n: int = 10,
) -> list[Restaurant]:
    """
    Main Phase 2 API:

    - Applies hard filters according to user preferences.
    - Scores remaining restaurants.
    - Sorts by score (descending).
    - Returns top N unique restaurants as `Restaurant` objects.
    """
    if df is None or df.empty:
        return []


    filtered = _filter_by_preferences(df, preferences)
    if filtered.empty:
        return []

    working = filtered.copy()
    working["__score"] = working.apply(
        lambda row: _compute_score(row, preferences),
        axis=1,
    )

    # Sort by score descending
    working = working.sort_values("__score", ascending=False)

    # Extra safety: remove any duplicates that might still exist in the processed dataset
    dedup_subset = []
    if "name" in working.columns:
        dedup_subset.append("name")
    loc_col = _detect_location_column(working)
    if loc_col:
        dedup_subset.append(loc_col)

    if dedup_subset:
        working = working.drop_duplicates(subset=dedup_subset, keep="first")

    top = working.head(top_n)

    restaurants: list[Restaurant] = []
    for _, row in top.iterrows():
        cuisines = row.get("cuisines_clean")
        if isinstance(cuisines, list):
            cuisines_list = cuisines
        elif isinstance(cuisines, str):
            cuisines_list = [c.strip() for c in cuisines.split(",") if c.strip()]
        else:
            cuisines_list = []

        name = row.get("name") or row.get("restaurant_name") or ""
        location_val = (
            row.get("location")
            or row.get("listed_in(city)")
            or row.get("city")
            or row.get("address")
        )

        restaurants.append(
            Restaurant(
                name=name,
                location=location_val,
                rating=row.get("rating")
                if "rating" in row.index
                else None,
                price=row.get("price")
                if "price" in row.index
                else None,
                cuisines=cuisines_list,
            )
        )

    return restaurants


__all__ = [
    "UserPreference",
    "Restaurant",
    "get_candidate_restaurants",
]

