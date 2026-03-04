from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

try:
    import google.genai as genai
except ImportError:
    genai = None  # type: ignore[assignment]

# Default Gemini model
GEMINI_MODEL_NAME_DEFAULT = "models/gemini-2.5-flash"


@dataclass
class GeminiClient:
    """
    Thin wrapper around Gemini LLM using google.genai SDK.
    Expects GEMINI_API_KEY in environment or .env.
    """

    model_name: str = GEMINI_MODEL_NAME_DEFAULT
    _client: Optional[genai.Client] = None

    def __post_init__(self) -> None:
        if genai is None:
            raise RuntimeError(
                "google-genai is not installed. Install with `pip install google-genai`."
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your environment or Streamlit Secrets."
            )

        self._client = genai.Client(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate content using Gemini and return plain text.
        """
        if self._client is None:
            raise RuntimeError("Gemini client not initialized.")

        # Gemini models now use `responses.create` for completions
        response = self._client.responses.create(
            model=self.model_name,
            # Provide prompts as a single string
            input=f"{system_prompt}\n\n{user_prompt}",
            temperature=0.7,
            max_output_tokens=500,
        )

        # Extract text from response safely
        try:
            return response.output_text  # modern genai attribute
        except AttributeError:
            return ""
