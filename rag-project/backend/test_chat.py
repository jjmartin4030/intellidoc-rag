"""Standalone test script for the chat endpoint.

Usage:
    python test_chat.py <doc_id>

Sends 3 hardcoded questions to POST /api/chat/ and prints the results.
The backend server must be running on http://127.0.0.1:8000.
"""

import sys
import httpx

API_URL = "http://127.0.0.1:8000/api/chat/"

TEST_QUESTIONS = [
    # 1. Likely IN the document
    "What is this document about?",
    # 2. Likely OUT of context
    "What is the capital of France?",
    # 3. Borderline
    "Summarize the main points",
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_chat.py <doc_id>")
        sys.exit(1)

    doc_id = sys.argv[1]
    print(f"Testing chat endpoint with doc_id: {doc_id}")
    print("=" * 70)

    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n🔹 Test {i}: {question}")
        print("-" * 70)

        try:
            resp = httpx.post(
                API_URL,
                json={"doc_id": doc_id, "question": question},
                timeout=60.0,
            )

            if resp.status_code != 200:
                print(f"   ❌ HTTP {resp.status_code}: {resp.json().get('detail', resp.text)}")
                print("=" * 70)
                continue

            data = resp.json()
            print(f"   is_out_of_context : {data['is_out_of_context']}")
            print(f"   top_score         : {data['top_score']}")
            print(f"   filename          : {data['filename']}")
            print(f"   source_chunks     : {len(data['source_chunks'])} chunks")
            print(f"   answer            :")
            print()
            # Indent the answer for readability
            for line in data["answer"].split("\n"):
                print(f"      {line}")

        except httpx.ConnectError:
            print("   ❌ Could not connect — is the server running on http://127.0.0.1:8000?")
        except Exception as exc:
            print(f"   ❌ Error: {exc}")

        print("=" * 70)


if __name__ == "__main__":
    main()
