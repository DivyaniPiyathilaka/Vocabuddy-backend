import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "words.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Add columns if they don't exist
    for sql in [
        "ALTER TABLE words ADD COLUMN source TEXT DEFAULT 'seed'",
        "ALTER TABLE words ADD COLUMN approved_by TEXT",
        "ALTER TABLE words ADD COLUMN approved_at TEXT",
    ]:
        try:
            cur.execute(sql)
        except sqlite3.OperationalError:
            pass

    # Add unique constraint via index
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_words_unique_word_si ON words(word_si)")

    conn.commit()
    conn.close()
    print("[OK] Migration complete.")

if __name__ == "__main__":
    main()
