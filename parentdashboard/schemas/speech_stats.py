"""Schemas for speech progress and accuracy statistics."""
from typing import List
from pydantic import BaseModel, Field


class PhonemeStat(BaseModel):
    """Accuracy statistics for a single phoneme/letter group."""

    phoneme: str = Field(..., description="Internal phoneme code (e.g. S, R, T, K, N)")
    label: str = Field(..., description="Display label for the Sinhala character(s)")
    accuracy: float = Field(..., description="Accuracy percentage for this phoneme group")
    total_words: int = Field(..., description="Total words analysed for this phoneme group")
    correct_words: int = Field(..., description="Correctly pronounced words for this phoneme group")


class WeeklyProgressPoint(BaseModel):
    """Accuracy for a single day/week bucket."""

    date: str = Field(..., description="ISO date (YYYY-MM-DD)")
    accuracy: float = Field(..., description="Accuracy percentage for this date")
    total_words: int = Field(..., description="Total words analysed for this date")
    correct_words: int = Field(..., description="Correctly pronounced words for this date")


class SpeechStatsResponse(BaseModel):
    """Aggregated speech statistics returned to the frontend."""

    overall_accuracy: float = Field(..., description="Overall accuracy across all records")
    total_words: int = Field(..., description="Total number of words analysed")
    total_correct: int = Field(..., description="Total number of correctly pronounced words")
    phoneme_breakdown: List[PhonemeStat] = Field(
        ..., description="Accuracy statistics per phoneme/letter group"
    )
    weekly_progress: List[WeeklyProgressPoint] = Field(
        ..., description="Accuracy trend over time for charts"
    )
    monthly_session_count: int = Field(
        0,
        description="Number of practice records from the 1st of the current month to now",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "overall_accuracy": 76.5,
                "total_words": 120,
                "total_correct": 92,
                "phoneme_breakdown": [
                    {
                        "phoneme": "S",
                        "label": "ස",
                        "accuracy": 82.0,
                        "total_words": 30,
                        "correct_words": 25,
                    }
                ],
                "weekly_progress": [
                    {
                        "date": "Week 10, 2026",
                        "accuracy": 70.0,
                        "total_words": 20,
                        "correct_words": 14,
                    }
                ],
                "monthly_session_count": 42,
            }
        }


class ChildSummaryResponse(BaseModel):
    """Overview information for the parent dashboard main screen."""

    id: str = Field(..., description="Child identifier")
    name: str = Field(..., description="Child's display name")
    age: int = Field(..., description="Child's age in years")
    overall_accuracy: float = Field(
        ...,
        description="Accuracy from latest practice in each session (නිවැරදි බව)",
    )
    monthly_practice_count: int = Field(
        ...,
        description="Number of practices done in the current month (මෙම මාසයේ පුහුණු වාර)",
    )
    target_sounds: List[str] = Field(
        default_factory=list,
        description="Request letter from each session in the last 30 days (ඉලක්ක ශබ්ද)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "mock-child",
                "name": "එමා ජොන්සන්",
                "age": 7,
                "overall_accuracy": 78.0,
                "monthly_practice_count": 12,
                "target_sounds": ["s", "w", "cs", "ක"],
            }
        }


