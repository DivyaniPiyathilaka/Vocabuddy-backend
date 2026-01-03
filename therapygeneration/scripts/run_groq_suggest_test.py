import os
from therapygeneration.services.word_engine import PracticeService

# Make sure GROQ_API_KEY is set before running:
# PowerShell:
# $env:GROQ_API_KEY="your_key_here"

service = PracticeService()

# Use a case with missing words
preview = service.preview_activity(
    child_id="child_001",
    letter="ස",
    mode="starts_with",
    level=3,
    count=8
)

print("Preview missing_count:", preview["missing_count"])

if preview["can_generate"]:
    out = service.generate_suggestions(
        therapist_pin="1234",
        child_id="child_001",
        letter="ස",
        mode="starts_with",
        level=3,
        missing_count=preview["missing_count"],
        model="moonshotai/kimi-k2-instruct-0905"
    )
    print(out)
else:
    print("No generation needed.")
