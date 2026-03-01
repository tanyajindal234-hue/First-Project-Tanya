from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - handled by environment setup
    load_dataset = None  # type: ignore[assignment]

import gc


DATASET_NAME = "ManikaSaini/zomato-restaurant-recommendation"


def load_raw_zomato_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load the raw Zomato dataset from Hugging Face as a pandas DataFrame.

    This requires the `datasets` library to be installed and network access.
    """
    if load_dataset is None:
        raise RuntimeError(
            "datasets library is not installed. "
            "Install it with `pip install datasets` before running phase 1."
        )

    ds = load_dataset(DATASET_NAME, split=split)
    df = ds.to_pandas()
    
    # Memory optimization: Keep only necessary columns immediately
    essential_cols = [
        "name", "location", "cuisines", "approx_cost(for two people)", 
        "rate", "listed_in(city)", "restaurant_id", "url"
    ]
    existing_cols = [c for c in essential_cols if c in df.columns]
    df = df[existing_cols].copy()
    
    del ds
    gc.collect()
    return df


def _detect_price_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort detection of the price / cost column."""
    candidates: list[str] = []
    for col in df.columns:
        lower = col.lower()
        if "price" in lower or "cost" in lower or "approx" in lower:
            candidates.append(col)
    return candidates[0] if candidates else None


def _detect_rating_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort detection of the rating column."""
    for name in ("rating", "rate", "aggregate_rating"):
        for col in df.columns:
            if col.lower() == name:
                return col
    # Fallback: any column that clearly looks like rating
    for col in df.columns:
        if "rating" in col.lower() or col.lower() == "rate":
            return col
    return None


def _clean_price_value(value) -> Optional[float]:
    """
    Normalize a single price value.

    - Remove thousands separators like "1,500" -> "1500".
    - Extract numeric portion from mixed strings.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    # Remove commas used as thousand separators
    text = text.replace(",", "")

    # Extract first numeric token
    match = re.search(r"(\d+(\.\d+)?)", text)
    if not match:
        return None

    number_text = match.group(1)
    try:
        return float(number_text)
    except ValueError:
        return None


def _clean_rating_value(value) -> Optional[float]:
    """
    Normalize a single rating value.

    Examples:
    - "4.1/5" -> 4.1
    - "4.1" -> 4.1
    - "NEW", "-" -> None
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.upper() in {"NEW", "-"}:
        return None

    # Remove any "/5" or similar suffix
    if "/" in text:
        text = text.split("/", 1)[0].strip()

    try:
        return float(text)
    except ValueError:
        return None


def _clean_cuisines_value(value) -> list[str]:
    """
    Normalize cuisines:

    - Split on commas.
    - Strip whitespace.
    - Drop empty tokens.
    """
    if pd.isna(value):
        return []

    text = str(value)
    parts = [part.strip() for part in text.split(",")]
    return [p for p in parts if p]


def _deduplicate_restaurants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate restaurants.

    Priority order for deduplication keys:
    - Explicit id columns if present.
    - Otherwise combination of name + location/address.
    """
    id_candidates: Iterable[str] = (
        "restaurant_id",
        "id",
        "rest_id",
        "url",
        "URL",
    )
    for col in id_candidates:
        if col in df.columns:
            return df.drop_duplicates(subset=[col], keep="first").reset_index(drop=True)

    subset: list[str] = []
    for name in ("name", "restaurant_name"):
        if name in df.columns:
            subset.append(name)
            break
    for loc in ("location", "address", "city", "listed_in(city)"):
        if loc in df.columns:
            subset.append(loc)
            break

    if subset:
        return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

    # Fallback: drop exact duplicate rows
    return df.drop_duplicates().reset_index(drop=True)


def clean_restaurant_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Phase 1 preprocessing rules to a raw restaurant DataFrame.

    - Price: convert to numeric without thousand separators (e.g. 1500, not "1,500").
    - Rating: store as numeric float (e.g. 4.1, not "4.1/5").
    - Cuisines: store as a clean list of cuisines.
    - Remove duplicate restaurants.
    """
    df = df.copy()

    # Price normalization
    price_col = _detect_price_column(df)
    if price_col:
        df["price"] = df[price_col].map(_clean_price_value)

    # Rating normalization
    rating_col = _detect_rating_column(df)
    if rating_col:
        df["rating"] = df[rating_col].map(_clean_rating_value)

    # Cuisines normalization
    if "cuisines" in df.columns:
        df["cuisines_clean"] = df["cuisines"].map(_clean_cuisines_value)

    # Deduplicate restaurants
    df = _deduplicate_restaurants(df)
    
    gc.collect()
    return df


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """
    Save the processed dataset to disk (Parquet by default).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def run_phase1_preprocessing(
    split: str = "train",
    output_path: Path | str = Path("data/processed/zomato_clean.parquet"),
) -> Path:
    """
    Convenience function to run the full Phase 1 pipeline:

    - Load raw dataset from Hugging Face.
    - Apply cleaning rules.
    - Save processed dataset locally.
    """
    raw_df = load_raw_zomato_dataset(split=split)
    clean_df = clean_restaurant_dataframe(raw_df)
    
    del raw_df
    gc.collect()
    
    path = save_processed_dataset(clean_df, output_path)
    
    del clean_df
    gc.collect()
    
    return path


__all__ = [
    "load_raw_zomato_dataset",
    "clean_restaurant_dataframe",
    "save_processed_dataset",
    "run_phase1_preprocessing",
]

