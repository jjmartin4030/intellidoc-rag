import re
import unicodedata


def extract_text(file_path: str, file_type: str) -> str:
    """Extract plain text from a PDF or DOCX file.

    Args:
        file_path: Path to the document file.
        file_type: Either "pdf" or "docx".

    Returns:
        Cleaned plain-text string.

    Raises:
        ValueError: If file_type is unsupported or the document is empty/unreadable.
        RuntimeError: If the underlying library fails during extraction.
    """
    file_type = file_type.lower().strip()

    if file_type == "pdf":
        raw_text = _extract_pdf(file_path)
    elif file_type == "docx":
        raw_text = _extract_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

    cleaned = _clean_text(raw_text)

    if len(cleaned) < 50:
        raise ValueError("Document appears to be empty or unreadable")

    return cleaned


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_pdf(file_path: str) -> str:
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract text: {exc}") from exc


def _extract_docx(file_path: str) -> str:
    try:
        from docx import Document

        doc = Document(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        return "\n\n".join(paragraphs)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract text: {exc}") from exc


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    # 1. Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # 2. Remove null bytes and non-printable characters (keep \n and \t)
    text = re.sub(r"[^\S\n\t]", " ", text)          # normalize whitespace chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # 3. Collapse multiple consecutive blank lines into a single \n\n
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Strip leading/trailing whitespace
    text = text.strip()

    return text
