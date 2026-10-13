
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "chat_memory.db"


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            UNIQUE(user_id, fact)
        )
    """)
    return conn


def get_recent_messages(user_id, limit=10):
    with connect() as conn:
        rows = conn.execute(
            """SELECT role, content FROM messages
               WHERE user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()

    return [
        {"role": role, "content": content}
        for role, content in reversed(rows)
    ]


def save_message(user_id, role, content):
    with connect() as conn:
        conn.execute(
            """INSERT INTO messages (user_id, role, content)
               VALUES (?, ?, ?)""",
            (user_id, role, content),
        )


def get_long_term_memory(user_id):
    with connect() as conn:
        rows = conn.execute(
            "SELECT fact FROM memories WHERE user_id = ?",
            (user_id,),
        ).fetchall()

    return [row[0] for row in rows]


def remember(user_id, fact):
    fact = fact.strip()
    if not fact or len(fact) > 300:
        return False

    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO memories (user_id, fact) VALUES (?, ?)",
            (user_id, fact),
        )
    return True
