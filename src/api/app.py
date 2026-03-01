from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import logging
from src.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedRestaurantOut,
)
from src.core.recommendation_engine import (
    UserPreference,
    get_candidate_restaurants,
)
from src.llm.orchestrator import generate_llm_recommendations


DEFAULT_DATA_PATH = Path("data/processed/zomato_clean.parquet")


def load_processed_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the processed dataset produced by Phase 1.

    Raises HTTPException if the file is missing so the API can return a
    meaningful error instead of a low-level stack trace.
    """
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Processed dataset not found. "
                "Please run Phase 1 preprocessing to generate "
                f"{path.as_posix()}."
            ),
        )
    return pd.read_parquet(path)


def get_dataset() -> pd.DataFrame:
    """
    FastAPI dependency to provide the processed dataset.

    Tests can override this dependency to inject a synthetic dataset.
    """
    return load_processed_dataset()


def create_app(
    dataset_dependency: Callable[[], pd.DataFrame] | None = None,
) -> FastAPI:
    """
    Application factory primarily to make testing easy.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    app = FastAPI(title="AI Restaurant Recommendation Service")

    # Templates for the Phase 5 UI
    templates_dir = Path(__file__).resolve().parent.parent / "ui" / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    # Use DI for dataset loading so tests can override it easily.
    def _dataset_dep() -> pd.DataFrame:
        return (dataset_dependency or get_dataset)()

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        """
        Phase 5 UI: Zomato-inspired React search page.
        """
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/health")
    def health_check():
        """
        Phase 6: Health check endpoint for monitoring.
        """
        return {"status": "healthy", "version": "1.0.0"}

    @app.get("/api/v1/places", response_model=List[str])
    def list_places(df: pd.DataFrame = Depends(_dataset_dep)) -> List[str]:
        """
        Return the list of distinct place/location values from the dataset.
        Used by the UI for the location dropdown.
        """
        # Detect likely location column
        loc_col = None
        for candidate in ("location", "listed_in(city)", "city", "address"):
            if candidate in df.columns:
                loc_col = candidate
                break

        if not loc_col:
            return []

        series = df[loc_col].dropna().astype(str).str.strip()
        unique_places = sorted({s for s in series if s})
        return unique_places

    @app.post(
        "/api/v1/recommendations",
        response_model=RecommendationResponse,
    )
    def recommend(
        payload: RecommendationRequest,
        df: pd.DataFrame = Depends(_dataset_dep),
    ) -> RecommendationResponse:
        """
        Phase 3 endpoint: non-LLM restaurant recommendations.
        """
        prefs = UserPreference(
            location=payload.place,
            min_rating=payload.rating,
            cuisines=payload.cuisines,
            min_price=payload.min_price,
            max_price=payload.max_price,
        )

        candidates = get_candidate_restaurants(df, prefs, top_n=10)

        restaurants_out = [
            RecommendedRestaurantOut(
                name=r.name,
                location=r.location,
                rating=r.rating,
                price=r.price,
                cuisines=r.cuisines,
            )
            for r in candidates
        ]

        return RecommendationResponse(recommendations=restaurants_out)

    @app.post(
        "/api/v1/recommendations/ai",
        response_model=RecommendationResponse,
    )
    def recommend_ai(
        payload: RecommendationRequest,
        df: pd.DataFrame = Depends(_dataset_dep),
    ) -> RecommendationResponse:
        """
        Integrated Phase 3 + Phase 4 endpoint: AI-powered recommendations.
        """
        logger.info(f"AI recommendation request for place: {payload.place}")
        prefs = UserPreference(
            location=payload.place,
            min_rating=payload.rating,
            cuisines=payload.cuisines,
            min_price=payload.min_price,
            max_price=payload.max_price,
        )

        # 1. Get candidates using Phase 2 core engine
        candidates = get_candidate_restaurants(df, prefs, top_n=10)

        # 2. Get AI rankings and reasons using Phase 4 LLM orchestrator
        if not candidates:
            return RecommendationResponse(recommendations=[])

        try:
            llm_results = generate_llm_recommendations(prefs, candidates, max_results=5)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to non-AI results if LLM fails
            llm_results = []

        if llm_results:
            restaurants_out = [
                RecommendedRestaurantOut(
                    name=r.name,
                    location=r.location,
                    rating=r.rating,
                    price=r.price,
                    cuisines=r.cuisines,
                    reason=r.reason,
                )
                for r in llm_results
            ]
        else:
            # Fallback or if LLM didn't return matches
            restaurants_out = [
                RecommendedRestaurantOut(
                    name=r.name,
                    location=r.location,
                    rating=r.rating,
                    price=r.price,
                    cuisines=r.cuisines,
                )
                for r in candidates[:5]
            ]

        return RecommendationResponse(recommendations=restaurants_out)

    return app


app = create_app()


__all__ = ["app", "create_app", "get_dataset", "load_processed_dataset"]

