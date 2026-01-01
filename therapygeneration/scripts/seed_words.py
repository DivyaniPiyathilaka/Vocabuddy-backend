import sqlite3
import os

DB_PATH = os.path.join("data", "words.db")

# Sample Sinhala words containing letter "ස"
# difficulty: 1 = easy, 2 = medium, 3 = hard

WORDS = [
    ("සතු", 1, "animal"),
    ("සයුර", 1, "nature"),
    ("සමනලයා", 1, "animal"),
    ("සරංගලය", 2, "object"),
    ("සතුට", 1, "emotion"),
    ("සඳ", 1, "nature"),
    ("සිරි", 2, "name"),
    ("සතුන්", 2, "animal"),
    ("සමගිය", 2, "abstract"),
    ("සංකල්ප", 3, "abstract"),
    ("සමාජය", 2, "abstract"),
    ("සංවිධානය", 3, "abstract"),
    ("සංවේදනය", 3, "emotion"),
]

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        for word, difficulty, tags in WORDS:
            cur.execute(
                "INSERT INTO words (word_si, difficulty, tags, length) VALUES (?, ?, ?, ?)",
                (word, difficulty, tags, len(word))
            )

        conn.commit()
        print(f"[OK] Inserted {len(WORDS)} words into database.")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
