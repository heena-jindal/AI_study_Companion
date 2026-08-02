"""
tracking_service.py

Part 4's persistence layer (Q2/Q4 from the quiz): a SQLite database that
remembers quiz performance ACROSS requests -- something the LLM itself
can never do (Q1), since every API call to Groq starts with zero memory
of anything before it.

Kept in its own file, same reasoning as rag_service.py being separate
from llm_service.py -- this talks to SQLite, not to Groq or ChromaDB.
"""

import sqlite3
from datetime import datetime, timezone

DB_PATH = "study_companion.db"


def init_db():
    """
    Creates the quiz_attempts table if it doesn't exist yet. Called once
    when the app starts up. CREATE TABLE IF NOT EXISTS is safe to run
    every single time -- it won't wipe existing data or error if the
    table's already there from a previous run.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            question TEXT NOT NULL,
            difficulty TEXT,
            is_correct INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def record_attempt(topic: str, question: str, difficulty: str, is_correct: bool):
    """
    Writes ONE row -- this is the actual answer to Q6: this function is
    what /submit-answer calls to close the loop that /quiz alone never
    could. Every call to this function is one persisted memory the LLM
    itself doesn't have.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO quiz_attempts (topic, question, difficulty, is_correct, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            topic,
            question,
            difficulty,
            int(is_correct),  # SQLite has no true boolean type, stores as 0/1
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_weak_topics(min_attempts: int = 2) -> list:
    """
    Groups every recorded attempt by topic, computes accuracy per topic,
    and returns topics sorted WEAKEST first. `min_attempts` filters out
    topics you've only tried once or twice -- one lucky guess or one
    unlucky miss shouldn't count as a real "weak topic" yet.

    This is the actual payoff of the whole SQLite/persistence
    conversation -- turning raw stored rows into an answerable question:
    "what should I actually study more?"
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            topic,
            COUNT(*) AS total_attempts,
            SUM(is_correct) AS correct_attempts
        FROM quiz_attempts
        GROUP BY topic
        HAVING COUNT(*) >= ?
        ORDER BY (CAST(SUM(is_correct) AS FLOAT) / COUNT(*)) ASC
        """,
        (min_attempts,),
    ).fetchall()
    conn.close()

    return [
        {
            "topic": row["topic"],
            "total_attempts": row["total_attempts"],
            "correct_attempts": row["correct_attempts"],
            "accuracy": round(row["correct_attempts"] / row["total_attempts"], 2),
        }
        for row in rows
    ]