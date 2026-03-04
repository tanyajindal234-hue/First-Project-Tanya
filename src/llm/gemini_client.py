from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

try:
    import google.genai as genai
except ImportError:
    genai = None  # type: ignore

GEMINI_MODEL_NAME_DEFAULT = "models/gemini-2.5-flash"

@dataclass
class GeminiClient:
    """
    Thin wrapper around Gemini LLM using modern google-genai SDK.
    """
    model_name: str = GEMINI_MODEL_NAME_DEFAULT

    def __post_init__(self):
        if genai is None:
            raise RuntimeError(
                "google-genai is not installed. Install with `pip install google-genai`."
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your environment or Streamlit Secrets."
            )

        # Modern SDK: create a client with the API key
        self.client = genai.Client(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate text using Gemini (modern google-genai).
        """
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=f"{system_prompt}\n\n{user_prompt}",
                temperature=0.7,
                max_output_tokens=500,
            )
            return getattr(response, "output_text", "") or ""
        except Exception as e:
            print("Gemini AI error:", e)
            return ""
