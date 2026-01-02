from therapygeneration.services.word_engine import PracticeService

service = PracticeService()

print("\n--- Level 1 ---")
print(service.create_activity(
    child_id="child_001",
    letter="ය",
    mode="contains",
    level=2,
    count=4
))

print("\n--- Level 2 ---")
print(service.create_activity(
    child_id="child_001",
    letter="ස",
    mode="starts_with",
    level=2,
    count=4
))

print("\n--- Level 3 ---")
print(service.create_activity(
    child_id="child_001",
    letter="ස",
    mode="starts_with",
    level=3,
    count=4
))
