from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Optional
from src.core.recommendation_engine import Restaurant, UserPreference
from src.llm.gemini_client import GeminiClient

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
        "Only recommend restaurants from the provided candidate list.\n"
        "Return strictly as JSON."
    )

def _build_user_prompt(preferences: UserPreference, candidates: List[Restaurant], max_results: int) -> str:
    prefs_desc = {
        "location": preferences.location,
        "min_rating": preferences.min_rating,
        "cuisines": preferences.normalized_cuisines(),
        "max_price": preferences.max_price
    }
    candidates_payload = [
        {"index": i, "name": r.name, "location": r.location, "rating": r.rating, "price": r.price, "cuisines": r.cuisines}
        for i, r in enumerate(candidates)
    ]
    return (
        f"User preferences:\n{json.dumps(prefs_desc)}\n"
        f"Candidate restaurants:\n{json.dumps(candidates_payload)}\n"
        f"Select up to {max_results} and return JSON with index and reason."
    )

def _parse_llm_response(raw_text: str, candidates: List[Restaurant], max_results: int) -> List[LLMRecommendation]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    recommendations: List[LLMRecommendation] = []
    seen = set()
    for item in data:
        idx = item.get("index")
        reason = item.get("reason", "")
        if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            r = candidates[idx]
            recommendations.append(LLMRecommendation(r.name, r.location, r.rating, r.price, r.cuisines, reason))
        if len(recommendations) >= max_results:
            break
    return recommendations

def generate_llm_recommendations(preferences: UserPreference, candidates: List[Restaurant], client: Optional[GeminiClient] = None, max_results: int = 5) -> List[LLMRecommendation]:
    if not candidates:
        return []
    client = client or GeminiClient()
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(preferences, candidates, max_results)
    raw_text = client.generate(system_prompt, user_prompt)
    return _parse_llm_response(raw_text, candidates, max_results)
