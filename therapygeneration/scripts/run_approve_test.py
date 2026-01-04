from therapygeneration.services.word_engine import PracticeService

service = PracticeService()

# therapist selects some valid candidates from Groq output
selected = ["සන්සුන්", "සඳලු", "සෙරි", "සන්සුන්කම", "සන්සුන්කර"]

out = service.approve_words(
    therapist_pin="1234",
    child_id="child_001",
    letter="ස",
    mode="starts_with",
    level=3,
    requested_count=8,
    approved_words=selected,
    difficulty_mode="auto",
    tag_value="unclassified",
)

print(out)
