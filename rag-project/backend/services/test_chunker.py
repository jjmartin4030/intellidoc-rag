"""Standalone test script for the chunker.

Usage:
    python test_chunker.py <file_path> <file_type>

Example:
    python test_chunker.py sample.pdf pdf
    python test_chunker.py report.docx docx
"""

import sys
import json
import uuid


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_chunker.py <file_path> <file_type>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_type = sys.argv[2]

    from extractor import extract_text
    from chunker import chunk_document

    # Extract
    try:
        text = extract_text(file_path, file_type)
    except (ValueError, RuntimeError) as exc:
        print(f"EXTRACTION ERROR: {exc}")
        sys.exit(1)

    # Chunk
    doc_id = str(uuid.uuid4())
    filename = file_path.split("\\")[-1].split("/")[-1]

    try:
        chunks = chunk_document(text, doc_id, filename, file_type)
    except ValueError as exc:
        print(f"CHUNKING ERROR: {exc}")
        sys.exit(1)

    # --- Stats ---
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(enc.encode(c["text"])) for c in chunks]

    print(f"Total chunks: {len(chunks)}")
    print(f"Min tokens:   {min(token_counts)}")
    print(f"Max tokens:   {max(token_counts)}")
    print(f"Avg tokens:   {sum(token_counts) / len(token_counts):.1f}")

    # --- First 2 chunks ---
    print("\n" + "=" * 60)
    print("FIRST 2 CHUNKS")
    print("=" * 60)
    for chunk in chunks[:2]:
        print(json.dumps(chunk, indent=2))
        print("-" * 60)

    # --- Last chunk ---
    print("\n" + "=" * 60)
    print("LAST CHUNK")
    print("=" * 60)
    print(json.dumps(chunks[-1], indent=2))


if __name__ == "__main__":
    main()
