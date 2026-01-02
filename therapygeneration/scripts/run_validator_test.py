from therapygeneration.validators.sinhala_validator import validate_candidate_list
from therapygeneration.domain.level_rules import get_constraints

# Simulate therapist request
letter = "ස"
mode = "starts_with"
level = 1

constraints = get_constraints(level)

# Some test candidates (mix valid/invalid)
candidates = [
    "සතු",          # valid
    "සමනලයා",      # valid (but may be too long if you reduce max_len)
    "අම්මා",        # wrong letter/mode
    "sat",          # non-sinhala
    "සතු",          # duplicate
    "  සයුර  ",      # valid (whitespace)
    "",             # empty
]

results = validate_candidate_list(
    candidates,
    letter=letter,
    mode=mode,
    max_len=constraints.max_length,
    existing_words_normalized=None,  # later we pass DB words
)

print(f"Letter={letter}, mode={mode}, level={level}, max_len={constraints.max_length}")
for r in results:
    print({"word": r.word, "normalized": r.normalized, "valid": r.valid, "reasons": r.reasons})
