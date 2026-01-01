import sqlite3
import os
from pathlib import Path

# DB_PATH = os.path.join("data", "words.db")
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "words.db"

class WordRepository:
    def __init__(self):
        self.db_path = DB_PATH

    def get_words(
        self,
        letter: str,
        mode: str,
        difficulty: int,
        count: int
    ):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # base query
        query = "SELECT word_si FROM words WHERE difficulty = ?"
        params = [difficulty]

        # mode logic
        if mode == "contains":
            query += " AND word_si LIKE ?"
            params.append(f"%{letter}%")
        elif mode == "starts_with":
            query += " AND word_si LIKE ?"
            params.append(f"{letter}%")
        elif mode == "ends_with":
            query += " AND word_si LIKE ?"
            params.append(f"%{letter}")
        else:
            conn.close()
            raise ValueError("Invalid mode")

        query += " LIMIT ?"
        params.append(count)

        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        return [r[0] for r in rows]
