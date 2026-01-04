from typing import Dict, List

from therapygeneration.repository.word_engine import WordRepository
from therapygeneration.domain.level_rules import get_constraints
from therapygeneration.validators.sinhala_validator import validate_candidate_list
from therapygeneration.llm.groq_client import GroqClient


class PracticeService:
    def __init__(self):
        self.repo = WordRepository()

    def create_activity(self, child_id: str, letter: str, mode: str, level: int, count: int) -> Dict:
        return self.preview_activity(child_id=child_id, letter=letter, mode=mode, level=level, count=count)

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

        want = max(oversample, missing_count)

        raw_candidates = groq.suggest_words(
            letter=letter,
            mode=mode,
            count=want,
            max_len=constraints.max_length
        )

        #Block duplicates that already exist in DB
        existing_norm = self.repo.get_all_words_normalized()

        results = validate_candidate_list(
            raw_candidates,
            letter=letter,
            mode=mode,
            max_len=constraints.max_length,
            existing_words_normalized=existing_norm
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

    #Step 5: therapist approves selected words -> insert into DB
    def approve_words(
        self,
        *,
        therapist_pin: str,
        child_id: str,
        letter: str,
        mode: str,
        level: int,
        requested_count: int,
        approved_words: List[str],
        difficulty_mode: str = "auto",     # "auto" or "manual"
        manual_difficulty: int = 1,
        tag_value: str = "unclassified"    # simple now
    ) -> Dict:
        if therapist_pin != "1234":
            return {"ok": False, "error": "Invalid therapist PIN."}

        constraints = get_constraints(level)

        # Existing DB words to prevent duplicates
        existing_norm = self.repo.get_all_words_normalized()

        # Re-validate therapist selected words deterministically
        results = validate_candidate_list(
            approved_words,
            letter=letter,
            mode=mode,
            max_len=constraints.max_length,
            existing_words_normalized=existing_norm
        )

        valid_selected = [r.normalized for r in results if r.valid and r.normalized]
        invalid_selected = [{"word": r.word, "reasons": r.reasons} for r in results if not r.valid]

        # Difficulty assignment (simple + controllable)
        if difficulty_mode == "auto":
            difficulty_to_store = constraints.difficulty_range[1]
        else:
            difficulty_to_store = int(manual_difficulty)

        inserted: List[str] = []
        skipped_existing: List[str] = []

        for w in valid_selected:
            ok = self.repo.insert_word(
                word_si=w,
                difficulty=difficulty_to_store,
                tags=tag_value,
                source="llm_approved",
                approved_by="therapist_demo",
            )
            if ok:
                inserted.append(w)
            else:
                skipped_existing.append(w)

        # Return updated preview for the original requested_count
        updated = self.preview_activity(
            child_id=child_id,
            letter=letter,
            mode=mode,
            level=level,
            count=requested_count
        )

        return {
            "ok": True,
            "inserted": inserted,
            "skipped_existing": skipped_existing,
            "invalid_selected": invalid_selected,
            "updated_preview": updated
        }
