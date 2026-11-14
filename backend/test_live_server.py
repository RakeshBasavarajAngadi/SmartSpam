"""
backend/test_live_server.py
---------------------------
Live HTTP integration test against a running Flask server instance.
Verifies network socket connectivity, HTML template delivery,
static asset serving, RESTful API prediction, stats, history, and deletion.
"""

import threading
import time
import urllib.request
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Use isolated test database for live integration tests
TEST_LIVE_DB_PATH = BASE_DIR / "backend" / "test_live_smartspam.db"
os.environ["SMARTSPAM_DB_PATH"] = str(TEST_LIVE_DB_PATH)

from werkzeug.serving import make_server
from backend.app import app


class ServerThread(threading.Thread):
    def __init__(self, app_instance, host='127.0.0.1', port=5000):
        super().__init__()
        self.server = make_server(host, port, app_instance)
        self.ctx = app_instance.app_context()
        self.ctx.push()

    def run(self):
        print("[LIVE TEST] Server started on http://127.0.0.1:5000")
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()
        print("[LIVE TEST] Server gracefully stopped")


def run_live_tests():
    srv = ServerThread(app)
    srv.start()
    time.sleep(1.5)

    base_url = "http://127.0.0.1:5000"

    try:
        # 1. Test GET /
        req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "SmartSpam" in html
            print("  [LIVE PASS] Live HTTP GET /: 200 OK (Served HTML UI)")

        # 2. Test GET /static/css/style.css
        with urllib.request.urlopen(f"{base_url}/static/css/style.css") as resp:
            css = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "--primary" in css
            print("  [LIVE PASS] Live HTTP GET /static/css/style.css: 200 OK")

        # 3. Test GET /static/js/app.js
        with urllib.request.urlopen(f"{base_url}/static/js/app.js") as resp:
            js = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "sampleMessages" in js
            print("  [LIVE PASS] Live HTTP GET /static/js/app.js: 200 OK")

        # 4. Test POST /api/predict (Spam)
        spam_data = json.dumps({"message": "WINNER! You won $10,000 cash. Call 9876543210 now!"}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/predict", data=spam_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 201
            assert res_json["classification"] == "Spam"
            record_id = res_json["id"]
            print(f"  [LIVE PASS] Live HTTP POST /api/predict: 201 Created (ID: {record_id}, {res_json['classification']}, {res_json['confidence']}%)")

        # 5. Test GET /api/stats
        with urllib.request.urlopen(f"{base_url}/api/stats") as resp:
            stats_json = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert stats_json["stats"]["total_analyzed"] >= 1
            print(f"  [LIVE PASS] Live HTTP GET /api/stats: 200 OK (Total: {stats_json['stats']['total_analyzed']})")

        # 6. Test GET /api/history
        with urllib.request.urlopen(f"{base_url}/api/history") as resp:
            hist_json = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert len(hist_json["history"]) >= 1
            print(f"  [LIVE PASS] Live HTTP GET /api/history: 200 OK (Found {len(hist_json['history'])} records)")

        # 7. Test DELETE /api/history/<id>
        del_req = urllib.request.Request(f"{base_url}/api/history/{record_id}", method="DELETE")
        with urllib.request.urlopen(del_req) as resp:
            del_json = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            print(f"  [LIVE PASS] Live HTTP DELETE /api/history/{record_id}: 200 OK")

        print("\n[ALL 7 LIVE SERVER INTEGRATION TESTS PASSED SUCCESSFULLY!]")

    finally:
        srv.shutdown()
        try:
            if TEST_LIVE_DB_PATH.exists():
                TEST_LIVE_DB_PATH.unlink()
                print(f"[CLEANUP] Successfully removed isolated live test database: {TEST_LIVE_DB_PATH}")
        except Exception as e:
            print(f"[CLEANUP WARNING] Could not remove live test database: {e}")


if __name__ == "__main__":
    run_live_tests()

