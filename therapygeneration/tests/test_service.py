from therapygeneration.services.word_engine import PracticeService

service = PracticeService()

result = service.create_activity(
    child_id="child_001",
    letter="ස",
    mode="starts_with",
    level=1,
    count=4
)

print(result)
