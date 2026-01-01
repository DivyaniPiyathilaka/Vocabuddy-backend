from therapygeneration.repository.word_engine import WordRepository

repo = WordRepository()

print(repo.get_words(letter="ස", mode="starts_with", difficulty=3, count=5))
print(repo.get_words(letter="ස", mode="contains", difficulty=2, count=5))
print(repo.get_words(letter="ස", mode="ends_with", difficulty=1, count=5))
