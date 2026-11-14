"""
backend/test_api.py
-------------------
Comprehensive test suite for SmartSpam Backend API & Database:
1. POST /api/predict (Valid spam, valid ham, empty message, oversize message, invalid JSON).
2. GET /api/history (Listing, pagination limit/offset, data format).
3. GET /api/stats (Zero records, post-insert metrics, percentage verification).
4. DELETE /api/history/<id> (Valid ID deletion, non-existent ID 404).
5. Database integrity and consistency validation.
"""

import sys
import os
import json
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Use isolated test database to protect production smartspam.db
TEST_DB_PATH = BASE_DIR / "backend" / "test_smartspam.db"
os.environ["SMARTSPAM_DB_PATH"] = str(TEST_DB_PATH)

from backend.app import create_app
from backend.database import get_db_connection, init_db


def run_tests():
    print("=" * 65)
    print("      SMARTSPAM: BACKEND API & DATABASE TEST SUITE")
    print("=" * 65)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    # Clean test state: clear database before tests
    with get_db_connection() as conn:
        conn.execute("DELETE FROM prediction_history;")
        conn.commit()

    passed_tests = 0
    total_tests = 0

    def assert_test(condition, test_name):
        nonlocal passed_tests, total_tests
        total_tests += 1
        if condition:
            passed_tests += 1
            print(f"  [PASS] {test_name}")
        else:
            print(f"  [FAIL] {test_name}")
            raise AssertionError(f"Test failed: {test_name}")

    # -------------------------------------------------------------
    # 1. Health Check Test
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 1: HEALTH CHECK ---")
    res = client.get("/api/health")
    assert_test(res.status_code == 200, "Health check returns 200 OK")
    assert_test(res.get_json().get("status") == "healthy", "Health status is 'healthy'")

    # -------------------------------------------------------------
    # 2. Stats on Empty DB
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 2: STATS ON EMPTY DATABASE ---")
    res = client.get("/api/stats")
    assert_test(res.status_code == 200, "Stats returns 200 OK")
    stats = res.get_json()["stats"]
    assert_test(stats["total_analyzed"] == 0, "Initial total_analyzed is 0")
    assert_test(stats["total_spam"] == 0, "Initial total_spam is 0")
    assert_test(stats["total_ham"] == 0, "Initial total_ham is 0")
    assert_test(stats["spam_percentage"] == 0.0, "Initial spam_percentage is 0.0")

    # -------------------------------------------------------------
    # 3. Predict Endpoint - Invalid / Malformed Input
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 3: INPUT VALIDATION & ERROR HANDLING ---")
    
    # 3.1 Non-JSON request
    res = client.post("/api/predict", data="plain text message")
    assert_test(res.status_code == 400, "Non-JSON body returns 400 Bad Request")

    # 3.2 Empty JSON
    res = client.post("/api/predict", json={})
    assert_test(res.status_code == 400, "Missing 'message' key returns 400 Bad Request")

    # 3.3 Empty string message
    res = client.post("/api/predict", json={"message": "   "})
    assert_test(res.status_code == 400, "Whitespace-only message returns 400 Bad Request")

    # 3.4 Oversized message (> 5000 chars)
    long_msg = "A" * 5001
    res = client.post("/api/predict", json={"message": long_msg})
    assert_test(res.status_code == 400, "Message exceeding 5000 chars returns 400 Bad Request")

    # -------------------------------------------------------------
    # 4. Predict Endpoint - Valid Predictions
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 4: PREDICTION & DATABASE PERSISTENCE ---")

    # 4.1 Valid Spam Message
    spam_payload = {
        "message": "WINNER!! As a valued network customer you have won a £1,000 cash prize! Call 09061743806 immediately to claim."
    }
    res_spam = client.post("/api/predict", json=spam_payload)
    assert_test(res_spam.status_code == 201, "Valid spam prediction returns 201 Created")
    data_spam = res_spam.get_json()
    assert_test(data_spam["classification"] == "Spam", "Correctly classified as 'Spam'")
    assert_test(data_spam["is_spam"] is True, "'is_spam' boolean is True")
    assert_test(data_spam["confidence"] >= 95.0, f"Spam confidence is high ({data_spam['confidence']}%)")
    spam_record_id = data_spam["id"]
    assert_test(isinstance(spam_record_id, int), "Returns integer record ID")

    # 4.2 Valid Legitimate (Ham) Message
    ham_payload = {
        "message": "Hey Mom, I will be home by 6 PM for dinner. Do you need me to pick up groceries on the way?"
    }
    res_ham = client.post("/api/predict", json=ham_payload)
    assert_test(res_ham.status_code == 201, "Valid ham prediction returns 201 Created")
    data_ham = res_ham.get_json()
    assert_test(data_ham["classification"] == "Ham", "Correctly classified as 'Ham'")
    assert_test(data_ham["is_spam"] is False, "'is_spam' boolean is False")
    assert_test(data_ham["confidence"] >= 90.0, f"Ham confidence is high ({data_ham['confidence']}%)")
    ham_record_id = data_ham["id"]

    # 4.3 Second Spam Message
    spam2_payload = {
        "message": "URGENT! Your banking access has been limited. Verify account details at http://secure-bank.login.com"
    }
    res_spam2 = client.post("/api/predict", json=spam2_payload)
    assert_test(res_spam2.status_code == 201, "Second spam prediction returns 201 Created")
    data_spam2 = res_spam2.get_json()
    assert_test(data_spam2["classification"] == "Spam", "Correctly classified as 'Spam'")
    spam2_record_id = data_spam2["id"]

    # -------------------------------------------------------------
    # 5. History Endpoint
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 5: HISTORY RETRIEVAL & PAGINATION ---")
    res_hist = client.get("/api/history")
    assert_test(res_hist.status_code == 200, "History returns 200 OK")
    hist_data = res_hist.get_json()
    assert_test(hist_data["count"] == 3, f"History contains exactly 3 records (got {hist_data['count']})")
    assert_test(hist_data["history"][0]["id"] == spam2_record_id, "History is ordered newest first")

    # Test pagination: limit=2
    res_limit = client.get("/api/history?limit=2&offset=0")
    limit_data = res_limit.get_json()
    assert_test(limit_data["count"] == 2, "Pagination limit=2 returns 2 records")

    # -------------------------------------------------------------
    # 6. Dashboard Stats Consistency
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 6: STATS CONSISTENCY VERIFICATION ---")
    res_stats = client.get("/api/stats")
    assert_test(res_stats.status_code == 200, "Stats returns 200 OK")
    stats_data = res_stats.get_json()["stats"]
    assert_test(stats_data["total_analyzed"] == 3, "Total analyzed is 3")
    assert_test(stats_data["total_spam"] == 2, "Total spam is 2")
    assert_test(stats_data["total_ham"] == 1, "Total ham is 1")
    expected_pct = round((2 / 3) * 100, 2)
    assert_test(stats_data["spam_percentage"] == expected_pct, f"Spam percentage matches calculation ({expected_pct}%)")

    # -------------------------------------------------------------
    # 7. Delete Endpoint
    # -------------------------------------------------------------
    print("\n--- TEST GROUP 7: RECORD DELETION ---")
    # Delete the second spam record
    res_del = client.delete(f"/api/history/{spam2_record_id}")
    assert_test(res_del.status_code == 200, "Deleting existing record returns 200 OK")

    # Try deleting the same record again (404)
    res_del_404 = client.delete(f"/api/history/{spam2_record_id}")
    assert_test(res_del_404.status_code == 404, "Deleting non-existent record returns 404 Not Found")

    # Verify updated stats post-deletion
    res_post_del = client.get("/api/stats")
    post_stats = res_post_del.get_json()["stats"]
    assert_test(post_stats["total_analyzed"] == 2, "Total analyzed decreased to 2 after deletion")
    assert_test(post_stats["total_spam"] == 1, "Total spam decreased to 1")
    assert_test(post_stats["total_ham"] == 1, "Total ham remains 1")
    assert_test(post_stats["spam_percentage"] == 50.0, "Spam percentage updated to 50.0%")

    print("\n" + "=" * 65)
    print(f"ALL {passed_tests}/{total_tests} TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

    # Clean up isolated test database
    try:
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
            print(f"[CLEANUP] Successfully removed isolated test database: {TEST_DB_PATH}")
    except Exception as e:
        print(f"[CLEANUP WARNING] Could not remove test database: {e}")


if __name__ == "__main__":
    run_tests()

