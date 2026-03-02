import sqlite3
from pathlib import Path
from typing import Optional, Set
from datetime import datetime

from therapygeneration.validators.sinhala_validator import normalize_si

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "words.db"


class WordRepository:
    def __init__(self):
        self.db_path = DB_PATH

    def get_words(self, letter: str, mode: str, difficulty: int, count: int):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        query = "SELECT word_si FROM words WHERE difficulty = ?"
        params = [difficulty]

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

    #Step 5 helpers

    def word_exists(self, word_si: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM words WHERE word_si = ? LIMIT 1", (word_si,))
            return cur.fetchone() is not None
        finally:
            conn.close()

    def get_all_words_normalized(self) -> Set[str]:
        """
        Used to block inserting duplicates from LLM suggestions.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT word_si FROM words")
            rows = cur.fetchall()
            return {normalize_si(r[0]) for r in rows if r and r[0]}
        finally:
            conn.close()

    def insert_word(
        self,
        *,
        word_si: str,
        difficulty: int,
        tags: str,
        source: str = "llm_approved",
        approved_by: Optional[str] = "therapist_demo",
        approved_at: Optional[str] = None,
    ) -> bool:
        """
        Returns True if inserted, False if skipped because it already exists.
        Requires DB columns: source, approved_by, approved_at (migration script below).
        """
        if approved_at is None:
            approved_at = datetime.utcnow().isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO words (word_si, difficulty, tags, length, source, approved_by, approved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (word_si, difficulty, tags, len(word_si), source, approved_by, approved_at)
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # unique index prevented duplicate OR word already exists
                return False
        finally:
            conn.close()
