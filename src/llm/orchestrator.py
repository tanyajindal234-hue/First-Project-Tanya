from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Protocol

from src.core.recommendation_engine import Restaurant, UserPreference
from src.llm.gemini_client import GeminiClient


class GeminiClientProtocol(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - interface only
        ...


@dataclass
class LLMRecommendation:
    name: str
    location: Optional[str]
    rating: Optional[float]
    price: Optional[float]
    cuisines: List[str]
    reason: str


def _build_system_prompt() -> str:
    return (
        "You are an AI assistant that recommends restaurants.\n"
        "You are given a list of candidate restaurants and the user's preferences.\n"
        "Only recommend restaurants from the provided candidate list.\n"
        "Return your answer strictly as JSON, with no additional text."
    )


def _build_user_prompt(
    preferences: UserPreference,
    candidates: List[Restaurant],
    max_results: int,
) -> str:
    prefs_desc = {
        "location": preferences.location,
        "min_rating": preferences.min_rating,
        "cuisines": preferences.normalized_cuisines(),
        "min_price": preferences.min_price,
        "max_price": preferences.max_price,
    }

    candidates_payload = []
    for idx, r in enumerate(candidates):
        candidates_payload.append(
            {
                "index": idx,
                "name": r.name,
                "location": r.location,
                "rating": r.rating,
                "price": r.price,
                "cuisines": r.cuisines,
            }
        )

    example_json = [
        {
            "index": 0,
            "reason": "High rating and matches the user's cuisine and price preferences.",
        }
    ]

    return (
        "User preferences:\n"
        f"{json.dumps(prefs_desc, ensure_ascii=False, indent=2)}\n\n"
        "Candidate restaurants (each has an 'index' field you must use):\n"
        f"{json.dumps(candidates_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Select up to {max_results} of the best restaurants.\n"
        "Respond ONLY with a JSON array, where each item has:\n"
        '  - "index": integer index of the chosen candidate restaurant\n'
        '  - "reason": short explanation of why it is a good match\n\n'
        "Example response format (do NOT repeat this example restaurant, just follow the format):\n"
        f"{json.dumps(example_json, ensure_ascii=False, indent=2)}"
    )


def _parse_llm_response(
    raw_text: str,
    candidates: List[Restaurant],
    max_results: int,
) -> List[LLMRecommendation]:
    """
    Parse the JSON response from the LLM and map back to candidate restaurants.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # If parsing fails, return empty list and let the caller decide how to degrade.
        return []

    if not isinstance(data, list):
        return []

    recommendations: List[LLMRecommendation] = []
    seen_indices: set[int] = set()

    for item in data:
        if not isinstance(item, dict):
            continue

        idx = item.get("index")
        reason = item.get("reason") or ""
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(candidates):
            continue
        if idx in seen_indices:
            continue

        seen_indices.add(idx)
        restaurant = candidates[idx]

        recommendations.append(
            LLMRecommendation(
                name=restaurant.name,
                location=restaurant.location,
                rating=restaurant.rating,
                price=restaurant.price,
                cuisines=restaurant.cuisines,
                reason=reason,
            )
        )

        if len(recommendations) >= max_results:
            break

    return recommendations


def generate_llm_recommendations(
    preferences: UserPreference,
    candidates: List[Restaurant],
    client: Optional[GeminiClientProtocol] = None,
    max_results: int = 5,
) -> List[LLMRecommendation]:
    """
    Phase 4 orchestration function:

    - Builds a structured prompt from user preferences and Phase 2 candidates.
    - Calls Gemini through a thin client (injected for testability).
    - Parses the JSON response and returns structured LLMRecommendation objects.
    """
    if not candidates:
        return []

    gemini_client: GeminiClientProtocol = client or GeminiClient()

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(preferences, candidates, max_results)

    raw_text = gemini_client.generate(system_prompt, user_prompt)
    return _parse_llm_response(raw_text, candidates, max_results)


__all__ = [
    "LLMRecommendation",
    "generate_llm_recommendations",
]

