"""
backend/test_edge_cases.py
--------------------------
Comprehensive Edge Case and Robustness Test Suite for SmartSpam:
1. Empty messages
2. Whitespace-only messages
3. Very long messages (> 5000 characters)
4. Messages containing URLs
5. Messages containing phone numbers
6. Messages containing emojis
7. Messages containing unusual punctuation / symbols
8. Mixed-language / foreign phrases
9. Legitimate messages containing suspicious words ("free", "prize", "scam") with negation
10. Spam messages with obfuscated words
"""

import sys
import os
import json
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Use isolated test database
TEST_EDGE_DB = BASE_DIR / "backend" / "test_edge_smartspam.db"
os.environ["SMARTSPAM_DB_PATH"] = str(TEST_EDGE_DB)

from backend.app import create_app
from backend.predictor import get_predictor


def run_edge_case_tests():
    print("=" * 68)
    print("      SMARTSPAM: COMPREHENSIVE EDGE CASE TEST SUITE")
    print("=" * 68)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    predictor = get_predictor()

    passed = 0
    total = 0

    def record_test(success, name, details=""):
        nonlocal passed, total
        total += 1
        if success:
            passed += 1
            print(f"  [PASS] {name}")
            if details:
                print(f"         └─ {details}")
        else:
            print(f"  [FAIL] {name}")
            raise AssertionError(f"Edge case failed: {name}")

    # 1. Empty message
    print("\n--- TEST 1: EMPTY MESSAGE ---")
    res1 = client.post("/api/predict", json={"message": ""})
    record_test(res1.status_code == 400, "Empty message rejected with HTTP 400", res1.get_json()["message"])

    # 2. Whitespace-only message
    print("\n--- TEST 2: WHITESPACE-ONLY MESSAGE ---")
    res2 = client.post("/api/predict", json={"message": "   \n\t   "})
    record_test(res2.status_code == 400, "Whitespace message rejected with HTTP 400", res2.get_json()["message"])

    # 3. Very long message (> 5000 characters)
    print("\n--- TEST 3: VERY LONG MESSAGE (> 5000 CHARACTERS) ---")
    long_text = "This is a legitimate conversation that continues. " * 110  # ~5500 chars
    res3 = client.post("/api/predict", json={"message": long_text})
    record_test(res3.status_code == 400, "Oversized message (> 5000 chars) rejected with HTTP 400", res3.get_json()["message"])

    # 4. Message containing URLs
    print("\n--- TEST 4: MESSAGE CONTAINING URLS ---")
    url_spam = "Final notice! You have an unclaimed parcel. Visit https://delivery-tracking.parcel-auth.com/claim to confirm."
    res4 = client.post("/api/predict", json={"message": url_spam})
    data4 = res4.get_json()
    record_test(res4.status_code == 201 and data4["classification"] == "Spam", 
                "URL message normalized and classified as Spam",
                f"Confidence: {data4['confidence']}%, Tokens: {data4['cleaned_text']}")

    # 5. Message containing Phone Numbers
    print("\n--- TEST 5: MESSAGE CONTAINING PHONE NUMBERS ---")
    phone_spam = "URGENT! Call 09061743806 or text STOP to 87121 to claim your £1500 store credit."
    res5 = client.post("/api/predict", json={"message": phone_spam})
    data5 = res5.get_json()
    record_test(res5.status_code == 201 and data5["classification"] == "Spam",
                "Phone/numeric message normalized and classified as Spam",
                f"Confidence: {data5['confidence']}%, Tokens: {data5['cleaned_text']}")

    # 6. Messages containing Emojis
    print("\n--- TEST 6: MESSAGES CONTAINING EMOJIS ---")
    emoji_ham = "Hey Sarah! 🎉 Congrats on the new job! 🚀 Are we celebrating tonight? 🍻😊"
    res6 = client.post("/api/predict", json={"message": emoji_ham})
    data6 = res6.get_json()
    record_test(res6.status_code == 201 and data6["classification"] == "Ham",
                "Emoji message handled gracefully without crash and classified as Ham",
                f"Confidence: {data6['confidence']}%, Tokens: {data6['cleaned_text']}")

    # 7. Messages containing Unusual Punctuation
    print("\n--- TEST 7: UNUSUAL PUNCTUATION / SYMBOLS ---")
    punct_msg = "W!O!W??? $$$ >>> Free *** Cash Prize --- Call Now <<< !!!"
    res7 = client.post("/api/predict", json={"message": punct_msg})
    data7 = res7.get_json()
    record_test(res7.status_code == 201 and data7["classification"] == "Spam",
                "Unusual punctuation stripped safely and classified as Spam",
                f"Confidence: {data7['confidence']}%, Tokens: {data7['cleaned_text']}")

    # 8. Mixed-Language / Foreign Phrases
    print("\n--- TEST 8: MIXED-LANGUAGE TEXT ---")
    mixed_ham = "Hola amigo! Just checking if you received the conference agenda for next Monday. Merci beaucoup!"
    res8 = client.post("/api/predict", json={"message": mixed_ham})
    data8 = res8.get_json()
    record_test(res8.status_code == 201 and data8["classification"] == "Ham",
                "Mixed-language message processed without error and classified as Ham",
                f"Confidence: {data8['confidence']}%, Tokens: {data8['cleaned_text']}")

    # 9. Legitimate Message Containing Suspicious Words (Negation Preservation)
    print("\n--- TEST 9: LEGITIMATE MESSAGE WITH SUSPICIOUS WORDS & NEGATION ---")
    negation_ham = "Hey John, I did not win any lottery prize and I am not interested in free cash scams. Do not click that link."
    res9 = client.post("/api/predict", json={"message": negation_ham})
    data9 = res9.get_json()
    record_test(res9.status_code == 201 and data9["classification"] == "Ham",
                "Negation words preserved; message correctly classified as Ham despite containing 'free', 'prize', 'scams'",
                f"Confidence: {data9['confidence']}%, Tokens: {data9['cleaned_text']}")

    # 10. Spam Message with Obfuscated Words
    print("\n--- TEST 10: SPAM MESSAGE WITH OBFUSCATED WORDS ---")
    obfuscated_spam = "C.a.s.h p.r.i.z.e! You are selected to receive £1,000. Call 08452810075 immediately or text WIN."
    res10 = client.post("/api/predict", json={"message": obfuscated_spam})
    data10 = res10.get_json()
    record_test(res10.status_code == 201 and data10["classification"] == "Spam",
                "Obfuscated spam message successfully identified as Spam",
                f"Confidence: {data10['confidence']}%, Tokens: {data10['cleaned_text']}")

    print("\n" + "=" * 68)
    print(f"ALL {passed}/{total} EDGE CASE TESTS PASSED SUCCESSFULLY!")
    print("=" * 68)

    # Clean up test database
    try:
        if TEST_EDGE_DB.exists():
            TEST_EDGE_DB.unlink()
            print(f"[CLEANUP] Cleaned isolated edge case test database: {TEST_EDGE_DB}")
    except Exception as e:
        print(f"[CLEANUP WARNING] {e}")


if __name__ == "__main__":
    run_edge_case_tests()

