from therapygeneration.services.word_engine import PracticeService

service = PracticeService()

# Use a case that likely returns fewer than requested (Level 3 only returned 3 in your output)
result = service.preview_activity(
    child_id="child_001",
    letter="ස",
    mode="starts_with",
    level=3,
    count=5
)

print(result)
