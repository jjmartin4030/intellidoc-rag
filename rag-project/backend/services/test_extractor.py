"""Standalone test script for the text extractor.

Usage:
    python test_extractor.py <file_path> <file_type>

Example:
    python test_extractor.py sample.pdf pdf
    python test_extractor.py report.docx docx
"""

import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_extractor.py <file_path> <file_type>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_type = sys.argv[2]

    from extractor import extract_text

    try:
        text = extract_text(file_path, file_type)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    char_count = len(text)
    word_count = len(text.split())
    preview = text[:500]

    print(f"Total characters: {char_count}")
    print(f"Total words:      {word_count}")
    print("-" * 60)
    print(preview)


if __name__ == "__main__":
    main()
