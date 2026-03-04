from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:  # pragma: no cover - exercised indirectly
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled gracefully at runtime
    genai = None  # type: ignore[assignment]


GEMINI_MODEL_NAME_DEFAULT = "gemini-2.5-pro"


@dataclass
class GeminiClient:
    """
    Thin wrapper around the Gemini LLM.

    It expects an API key in the environment variable `GEMINI_API_KEY`
    (loaded from `.env` if present).
    """

    model_name: str = GEMINI_MODEL_NAME_DEFAULT
    _model: Optional["genai.GenerativeModel"] = None  # type: ignore[name-defined]

    def __post_init__(self) -> None:
        if genai is None:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Install it with `pip install google-generativeai`."
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to your environment or `.env` file."
            )

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a completion using Gemini and return the text.
        """
        if self._model is None:
            raise RuntimeError("Gemini client is not initialized correctly.")

        response = self._model.generate_content(
            [system_prompt, user_prompt],
        )
        return getattr(response, "text", "") or ""


__all__ = ["GeminiClient", "GEMINI_MODEL_NAME_DEFAULT"]

