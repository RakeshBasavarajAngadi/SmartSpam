"""
backend/database.py
-------------------
SQLite database access module for SmartSpam.
Manages the prediction_history table, parameterized CRUD operations,
and dashboard statistics calculations.
"""

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import get_database_path


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with dictionary-like row access."""
    db_path = get_database_path()
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initializes the SQLite database and ensures the prediction_history table exists.
    Uses parameterized DDL and creates indexes for efficient ordering and filtering.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                classification TEXT NOT NULL CHECK(classification IN ('Spam', 'Ham')),
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_timestamp 
            ON prediction_history (timestamp DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_classification 
            ON prediction_history (classification);
        """)
        conn.commit()
    print(f"[DATABASE] Initialized SQLite database at: {get_database_path()}")


def insert_prediction(
    message: str,
    classification: str,
    confidence: float,
    timestamp: Optional[str] = None
) -> int:
    """
    Inserts a new classification record using parameterized queries.
    Returns the newly generated record ID.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_history (message, classification, confidence, timestamp)
            VALUES (?, ?, ?, ?);
            """,
            (message, classification, float(confidence), timestamp)
        )
        conn.commit()
        return cursor.lastrowid


def get_history(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieves recent predictions ordered newest first.
    Uses pagination with limit and offset.
    """
    limit = max(1, min(limit, 500))  # bounds check: 1 to 500
    offset = max(0, offset)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, message, classification, confidence, timestamp
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ? OFFSET ?;
            """,
            (limit, offset)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_prediction_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single prediction record by primary key."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, message, classification, confidence, timestamp
            FROM prediction_history
            WHERE id = ?;
            """,
            (record_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_prediction(record_id: int) -> bool:
    """
    Deletes a prediction record by its ID using parameterized queries.
    Returns True if a record was deleted, False if no such record existed.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM prediction_history
            WHERE id = ?;
            """,
            (record_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def get_statistics() -> Dict[str, Any]:
    """
    Calculates aggregate summary statistics for the dashboard.
    Returns total analyzed, spam count, ham count, spam percentage,
    and average confidence.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) AS total FROM prediction_history;")
        total = cursor.fetchone()['total']

        if total == 0:
            return {
                "total_analyzed": 0,
                "total_spam": 0,
                "total_ham": 0,
                "spam_percentage": 0.0,
                "ham_percentage": 0.0,
                "average_confidence": 0.0
            }

        # Breakdown by classification
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN classification = 'Spam' THEN 1 ELSE 0 END) AS total_spam,
                SUM(CASE WHEN classification = 'Ham' THEN 1 ELSE 0 END) AS total_ham,
                AVG(confidence) AS avg_conf
            FROM prediction_history;
        """)
        row = cursor.fetchone()
        total_spam = row['total_spam'] or 0
        total_ham = row['total_ham'] or 0
        avg_conf = row['avg_conf'] or 0.0

        spam_pct = round((total_spam / total) * 100, 2)
        ham_pct = round((total_ham / total) * 100, 2)

        return {
            "total_analyzed": int(total),
            "total_spam": int(total_spam),
            "total_ham": int(total_ham),
            "spam_percentage": float(spam_pct),
            "ham_percentage": float(ham_pct),
            "average_confidence": round(float(avg_conf), 2)
        }


if __name__ == "__main__":
    print("Testing SQLite database operations...")
    init_db()
    
    # Test insert
    test_id = insert_prediction("Sample spam test message", "Spam", 99.8)
    print(f"Inserted record with ID: {test_id}")
    
    # Test retrieve
    rec = get_prediction_by_id(test_id)
    print(f"Retrieved record: {rec}")
    
    # Test stats
    stats = get_statistics()
    print(f"Current Stats: {stats}")
    
    # Test delete
    deleted = delete_prediction(test_id)
    print(f"Deleted test record (success={deleted})")
    
    # Verify deletion
    post_stats = get_statistics()
    print(f"Stats after deletion: {post_stats}")
