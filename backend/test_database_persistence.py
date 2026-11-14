"""
backend/test_database_persistence.py
------------------------------------
Database persistence and server restart test:
Verifies that prediction history and stats are retained across server restarts
using an isolated test database without touching production smartspam.db.
"""

import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

PERSIST_TEST_DB = BASE_DIR / "backend" / "test_persist_smartspam.db"
os.environ["SMARTSPAM_DB_PATH"] = str(PERSIST_TEST_DB)

from backend.app import create_app
from backend.database import get_db_connection


def test_persistence_across_restarts():
    print("=" * 68)
    print("      SMARTSPAM: DATABASE PERSISTENCE & RESTART TEST")
    print("=" * 68)

    # Clean existing test file if present
    if PERSIST_TEST_DB.exists():
        try:
            PERSIST_TEST_DB.unlink()
        except Exception:
            pass

    # Session 1: Start App Instance 1
    print("\n[SESSION 1] Starting initial Flask instance...")
    app1 = create_app()
    client1 = app1.test_client()

    # Insert 1 Spam and 1 Ham
    print("[SESSION 1] Submitting predictions...")
    res_s = client1.post("/api/predict", json={"message": "Free £1000 prize! Call 09061743806."})
    assert res_s.status_code == 201
    s_id = res_s.get_json()["id"]

    res_h = client1.post("/api/predict", json={"message": "Meeting at 4pm for project status."})
    assert res_h.status_code == 201
    h_id = res_h.get_json()["id"]

    # Check stats before shutdown
    stats1 = client1.get("/api/stats").get_json()["stats"]
    print(f"[SESSION 1] Stats before shutdown: {stats1}")
    assert stats1["total_analyzed"] == 2
    assert stats1["total_spam"] == 1
    assert stats1["total_ham"] == 1
    assert stats1["spam_percentage"] == 50.0

    print("[SESSION 1] Simulating server shutdown (closing app & connections)...")
    del client1
    del app1

    # Session 2: Start Fresh App Instance 2 (Simulating Server Restart)
    print("\n[SESSION 2] Simulating server restart (fresh Flask app instance)...")
    app2 = create_app()
    client2 = app2.test_client()

    print("[SESSION 2] Checking retained prediction history...")
    hist_res = client2.get("/api/history")
    assert hist_res.status_code == 200
    hist_data = hist_res.get_json()["history"]

    assert len(hist_data) == 2, f"Expected 2 records retained, found {len(hist_data)}"
    assert hist_data[0]["id"] == h_id, "Latest record retained"
    assert hist_data[1]["id"] == s_id, "Earlier record retained"
    print(f"  [PASS] Retained {len(hist_data)} prediction records across restart with identical IDs ({s_id}, {h_id})")

    # Check stats after restart
    stats2 = client2.get("/api/stats").get_json()["stats"]
    print(f"[SESSION 2] Stats after restart: {stats2}")
    assert stats2["total_analyzed"] == 2
    assert stats2["total_spam"] == 1
    assert stats2["total_ham"] == 1
    assert stats2["spam_percentage"] == 50.0
    print("  [PASS] Aggregate stats match perfectly before and after restart!")

    print("\n" + "=" * 68)
    print("DATABASE PERSISTENCE AND RESTART TESTS PASSED SUCCESSFULLY!")
    print("=" * 68)

    # Cleanup
    del client2
    del app2


if __name__ == "__main__":
    test_persistence_across_restarts()

