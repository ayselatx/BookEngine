import sqlite3

DB_PATH = "data/notes.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            source TEXT,
            rating INTEGER,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def add_note_to_db(text, source="note", rating=3, embedding=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO notes (text, source, rating, embedding) VALUES (?, ?, ?, ?)",
        (text, source, rating, embedding)
    )
    conn.commit()
    conn.close()

def get_all_notes():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT text, source, rating, embedding FROM notes")
    rows = cursor.fetchall()
    conn.close()
    return rows
