# therapygeneration/domain/level_rules.py
from dataclasses import dataclass
from typing import Optional, Set, Tuple


@dataclass(frozen=True)
class LevelConstraints:
    """
    Deterministic, explainable constraints for each therapy level.
    V1: Keep simple. Later you can refine.
    """
    level: int
    difficulty_range: Tuple[int, int]      # inclusive (min, max)
    max_length: int                        # max word length allowed
    allowed_tags: Optional[Set[str]] = None
    blocked_tags: Optional[Set[str]] = None


# V1 rules (simple + defensible)
LEVEL_RULES = {
    1: LevelConstraints(
        level=1,
        difficulty_range=(1, 1),
        max_length=6,
        allowed_tags={"animal", "object", "nature"},
        blocked_tags={"abstract", "emotion"},
    ),
    2: LevelConstraints(
        level=2,
        difficulty_range=(1, 2),
        max_length=8,
        allowed_tags={"animal", "object", "nature", "name", "emotion"},
        blocked_tags={"abstract"},  # optional
    ),
    3: LevelConstraints(
        level=3,
        difficulty_range=(1, 3),
        max_length=12,
        allowed_tags=None,          # allow all
        blocked_tags=None,
    ),
}


def get_constraints(level: int) -> LevelConstraints:
    """
    Return deterministic constraints for a therapy level.
    Raises ValueError for invalid levels.
    """
    if level not in LEVEL_RULES:
        raise ValueError(f"Invalid level: {level}. Allowed levels: {sorted(LEVEL_RULES.keys())}")
    return LEVEL_RULES[level]
