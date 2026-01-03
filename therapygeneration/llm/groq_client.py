import os
import json
from typing import List, Dict, Any

from groq import Groq


class GroqClient:
    """
    Groq SDK client.
    Env var required:
      GROQ_API_KEY
    """

    def __init__(self, model: str = "moonshotai/kimi-k2-instruct-0905"):
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in environment variables.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def suggest_words(self, *, letter: str, mode: str, count: int, max_len: int) -> List[str]:
        request_count = max(count, 20)

        prompt = (
            f"Generate {request_count} Sinhala single-word candidates.\n"
            f"Constraints: Sinhala script only; no numbers; no English; no duplicates; "
            f'mode="{mode}"; target="{letter}"; prefer length<={max_len}.\n'
            'Return ONLY JSON exactly like: {"candidates":["..."]}\n'
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return ONLY JSON. No reasoning. No explanations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )

        content = (completion.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Groq returned empty content.")

        # Try parse, fallback extract JSON block
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(f"Groq returned non-JSON:\n{content}")
            parsed = json.loads(content[start:end+1])

        candidates = parsed.get("candidates", [])
        if not isinstance(candidates, list):
            raise RuntimeError(f"Groq JSON missing 'candidates' list. Got: {parsed}")

        return [str(x).strip() for x in candidates if str(x).strip()]
