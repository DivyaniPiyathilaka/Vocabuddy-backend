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
from typing import Dict, List


class PracticeService:
    def __init__(self):
        self.repo = WordRepository()

    def create_activity(
        self,
        child_id: str,
        letter: str,
        mode: str,
        level: int,
        count: int
    ) -> Dict:
        words = self.repo.get_words(
            letter=letter,
            mode=mode,
            difficulty=level,
            count=count
        )

        return {
            "child_id": child_id,
            "activity_type": "phonological_practice",
            "level": level,
            "target_letter": letter,
            "position": mode,  # initial / medial / final
            "item_type": "single_word",
            "items": self._build_items(words)
        }

    def _build_items(self, words: List[str]) -> List[Dict]:
        return [
            {
                "text": word,
                "language": "si"
            }
            for word in words
        ]


