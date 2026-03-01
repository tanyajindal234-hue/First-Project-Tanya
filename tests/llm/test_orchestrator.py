from src.core.recommendation_engine import Restaurant, UserPreference
from src.llm.orchestrator import (
    LLMRecommendation,
    generate_llm_recommendations,
)


class FakeGeminiClient:
    """
    Fake Gemini client used for tests so no real API call is made.
    """

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_system_prompt = None
        self.last_user_prompt = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self._response_text


def test_generate_llm_recommendations_parses_and_maps_indices():
    candidates = [
        Restaurant(
            name="Spicy House",
            location="Indiranagar",
            rating=4.5,
            price=1500.0,
            cuisines=["North Indian", "Chinese"],
        ),
        Restaurant(
            name="Okay House",
            location="Indiranagar",
            rating=4.0,
            price=1400.0,
            cuisines=["Chinese"],
        ),
        Restaurant(
            name="Far Away Diner",
            location="Koramangala",
            rating=4.6,
            price=900.0,
            cuisines=["North Indian"],
        ),
    ]

    prefs = UserPreference(
        location="Indiranagar",
        min_rating=4.0,
        cuisines=["North Indian"],
        min_price=200.0,
        max_price=2000.0,
    )

    # LLM selects index 0 and 2, with a duplicate and an invalid index to be ignored
    fake_response = """
    [
      {"index": 0, "reason": "High rating and matches North Indian cuisine."},
      {"index": 2, "reason": "Good alternative nearby."},
      {"index": 2, "reason": "Duplicate should be ignored."},
      {"index": 99, "reason": "Invalid index should be ignored."}
    ]
    """
    client = FakeGeminiClient(fake_response)

    results = generate_llm_recommendations(
        preferences=prefs,
        candidates=candidates,
        client=client,
        max_results=5,
    )

    assert isinstance(results, list)
    assert all(isinstance(r, LLMRecommendation) for r in results)

    # Only indices 0 and 2 are valid and unique
    assert len(results) == 2
    assert results[0].name == "Spicy House"
    assert results[0].reason.startswith("High rating")
    assert results[1].name == "Far Away Diner"
    assert results[1].reason.startswith("Good alternative")

