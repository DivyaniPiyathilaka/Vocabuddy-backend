from typing import Dict, List, Optional

from therapygeneration.repository.word_engine import WordRepository
from therapygeneration.domain.level_rules import get_constraints
from therapygeneration.validators.sinhala_validator import validate_candidate_list
from therapygeneration.llm.groq_client import GroqClient


class PracticeService:
    def __init__(self):
        self.repo = WordRepository()

    # Keep this if your UI currently calls create_activity
    def create_activity(self, child_id: str, letter: str, mode: str, level: int, count: int) -> Dict:
        # For now, create_activity can just call preview_activity and return that.
        return self.preview_activity(child_id=child_id, letter=letter, mode=mode, level=level, count=count)

    # Step 3 (already working): deterministic preview + missing_count
    def preview_activity(self, child_id: str, letter: str, mode: str, level: int, count: int) -> Dict:
        constraints = get_constraints(level)
        difficulty_to_use = constraints.difficulty_range[1]

        words = self.repo.get_words(
            letter=letter,
            mode=mode,
            difficulty=difficulty_to_use,
            count=count
        )

        missing_count = max(0, count - len(words))

        return {
            "child_id": child_id,
            "activity_type": "phonological_practice",
            "level": level,
            "constraints": {
                "difficulty_range": constraints.difficulty_range,
                "max_length": constraints.max_length,
                "allowed_tags": sorted(list(constraints.allowed_tags)) if constraints.allowed_tags else None,
                "blocked_tags": sorted(list(constraints.blocked_tags)) if constraints.blocked_tags else None,
            },
            "target_letter": letter,
            "position": mode,
            "item_type": "single_word",
            "items": [{"text": w, "language": "si"} for w in words],
            "requested_count": count,
            "returned_count": len(words),
            "missing_count": missing_count,
            "can_generate": missing_count > 0
        }

    # Step 4: Groq suggest-only + deterministic validation + PIN gate
    def generate_suggestions(
        self,
        *,
        therapist_pin: str,
        child_id: str,
        letter: str,
        mode: str,
        level: int,
        missing_count: int,
        model: str = "moonshotai/kimi-k2-instruct-0905",
        oversample: int = 20,
    ) -> Dict:
        if therapist_pin != "1234":
            return {"ok": False, "error": "Invalid therapist PIN."}

        constraints = get_constraints(level)

        groq = GroqClient(model=model)

        # oversample so therapist has options; validator will reject bad ones
        want = max(oversample, missing_count)

        raw_candidates = groq.suggest_words(
            letter=letter,
            mode=mode,
            count=want,
            max_len=constraints.max_length
        )

        # For now we only check duplicates inside suggestions.
        # In Step 5 we will also block "already exists in DB".
        results = validate_candidate_list(
            raw_candidates,
            letter=letter,
            mode=mode,
            max_len=constraints.max_length,
            existing_words_normalized=None
        )

        return {
            "ok": True,
            "child_id": child_id,
            "target_letter": letter,
            "position": mode,
            "level": level,
            "max_length": constraints.max_length,
            "requested_missing": missing_count,
            "candidates": [
                {
                    "word": r.word,
                    "normalized": r.normalized,
                    "valid": r.valid,
                    "reasons": r.reasons
                }
                for r in results
            ]
        }
