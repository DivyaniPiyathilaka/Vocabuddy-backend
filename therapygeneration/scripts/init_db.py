import os
import sqlite3

DB_PATH = os.path.join("data", "words.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_si TEXT NOT NULL,
    difficulty INTEGER NOT NULL DEFAULT 1,
    tags TEXT,
    length INTEGER
);
"""

def main():
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        print(f"[OK] Database ready at: {DB_PATH}")
        print("[OK] Table 'words' created (if it didn't exist).")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
