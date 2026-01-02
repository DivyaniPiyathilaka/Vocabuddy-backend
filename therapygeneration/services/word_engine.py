# from word_engine.repository import WordRepository
#
# class PracticeService:
#     def __init__(self):
#         self.repo = WordRepository()
#
#     def create_activity(
#         self,
#         letter: str,
#         mode: str,
#         level: int,
#         count: int
#     ):
#         words = self.repo.get_words(
#             letter=letter,
#             mode=mode,
#             difficulty=level,
#             count=count
#         )
#
#         # build activity payload (UI-friendly)
#         return {
#             "target_letter": letter,
#             "mode": mode,
#             "level": level,
#             "type": "single_word",
#             "items": [{"text": w} for w in words]
#         }

from therapygeneration.repository.word_engine import WordRepository
from therapygeneration.domain.level_rules import get_constraints
from typing import Dict, List


# class PracticeService:
#     def __init__(self):
#         self.repo = WordRepository()
#
#     def create_activity(
#         self,
#         child_id: str,
#         letter: str,
#         mode: str,
#         level: int,
#         count: int
#     ) -> Dict:
#         words = self.repo.get_words(
#             letter=letter,
#             mode=mode,
#             difficulty=level,
#             count=count
#         )

        # return {
        #             "child_id": child_id,
        #             "activity_type": "phonological_practice",
        #             "level": level,
        #             "target_letter": letter,
        #             "position": mode,  # initial / medial / final
        #             "item_type": "single_word",
        #             "items": self._build_items(words)
        #         }

class PracticeService:
    def __init__(self):
         self.repo = WordRepository()

    def create_activity(self, child_id: str, letter: str, mode: str, level: int, count: int):
        constraints = get_constraints(level)

        # V1: repo API currently supports a single difficulty value.
        # To keep deterministic and minimal changes, use the max difficulty for the level.

        difficulty_to_use = constraints.difficulty_range[1]
        words = self.repo.get_words(
            letter=letter,
            mode=mode,
            difficulty=difficulty_to_use,
            count=count
        )

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
        }



    def _build_items(self, words: List[str]) -> List[Dict]:
        return [
            {
                "text": word,
                "language": "si"
            }
            for word in words
        ]


