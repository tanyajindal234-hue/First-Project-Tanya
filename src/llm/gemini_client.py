from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import google.genai as genai
except ImportError:
    genai = None

GEMINI_MODEL_NAME_DEFAULT = "models/gemini-2.5-flash"

@dataclass
class GeminiClient:
    model_name: str = GEMINI_MODEL_NAME_DEFAULT
    _client: Optional["genai.Client"] = None

    def __post_init__(self):
        if genai is None:
            raise RuntimeError("google-genai not installed. pip install google-genai")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        self._client = genai.Client(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self._client:
            raise RuntimeError("Gemini client not initialized.")
        response = self._client.responses.create(
            model=self.model_name,
            input=f"{system_prompt}\n{user_prompt}"
        )
        return getattr(response, "output_text", "") or ""
