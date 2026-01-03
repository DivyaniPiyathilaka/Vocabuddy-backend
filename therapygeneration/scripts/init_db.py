import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("data", "words.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_si TEXT NOT NULL,
    difficulty INTEGER NOT NULL DEFAULT 1,
    tags TEXT,
    length INTEGER,
    source TEXT DEFAULT 'seed',
    approved_by TEXT,
    approved_at TEXT
);
"""

MIGRATIONS = [
    "ALTER TABLE words ADD COLUMN source TEXT DEFAULT 'seed';",
    "ALTER TABLE words ADD COLUMN approved_by TEXT;",
    "ALTER TABLE words ADD COLUMN approved_at TEXT;",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_words_unique_word_si ON words(word_si);"
]

def column_exists(cursor, table, column):
    cursor.execute(f"PRAGMA table_info({table});")
    return any(col[1] == column for col in cursor.fetchall())

def main():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    try:
        cur = conn.cursor()

        # Create table if missing
        cur.execute(CREATE_TABLE_SQL)

        # Run safe migrations
        if not column_exists(cur, "words", "source"):
            cur.execute("ALTER TABLE words ADD COLUMN source TEXT DEFAULT 'seed';")

        if not column_exists(cur, "words", "approved_by"):
            cur.execute("ALTER TABLE words ADD COLUMN approved_by TEXT;")

        if not column_exists(cur, "words", "approved_at"):
            cur.execute("ALTER TABLE words ADD COLUMN approved_at TEXT;")

        # Unique index (safe to run multiple times)
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_words_unique_word_si ON words(word_si);"
        )

        conn.commit()
        print("[OK] Database ready and migrated")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
