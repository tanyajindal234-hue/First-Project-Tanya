from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, conlist, confloat


class RecommendationRequest(BaseModel):
    """Request payload for the recommendations endpoint."""

    place: Optional[str] = Field(
        None,
        description="City/area/location where the user wants to eat.",
    )
    rating: Optional[confloat(ge=0, le=5)] = Field(
        None,
        description="Minimum rating threshold (0–5).",
    )
    min_price: Optional[float] = Field(
        None,
        description="Minimum price (numeric, same units as dataset).",
    )
    max_price: Optional[float] = Field(
        None,
        description="Maximum price (numeric, same units as dataset).",
    )
    cuisines: conlist(str, min_length=0) = Field(default_factory=list, description="List of preferred cuisines.")


class RecommendedRestaurantOut(BaseModel):
    name: str
    location: Optional[str] = None
    rating: Optional[float] = None
    price: Optional[float] = None
    cuisines: List[str] = Field(default_factory=list)
    reason: Optional[str] = None



class RecommendationResponse(BaseModel):
    recommendations: List[RecommendedRestaurantOut]


__all__ = [
    "RecommendationRequest",
    "RecommendedRestaurantOut",
    "RecommendationResponse",
]

