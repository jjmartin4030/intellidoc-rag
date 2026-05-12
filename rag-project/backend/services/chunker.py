import tiktoken
from datetime import datetime, timezone

# Tiktoken encoder for OpenAI cl100k_base (used by GPT-3.5/4)
_encoder = tiktoken.get_encoding("cl100k_base")

# Chunking parameters
CHUNK_SIZE = 500       # max tokens per chunk
CHUNK_OVERLAP = 75     # token overlap between consecutive chunks
MIN_CHUNK_TOKENS = 20  # skip chunks smaller than this

# Splitting separators in priority order
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _token_count(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(_encoder.encode(text))


def _split_by_separator(text: str, separator: str) -> list[str]:
    """Split text by a separator, keeping the separator at the end of each piece."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # Re-attach the separator to every piece except the last
    result = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            result.append(part + separator)
        else:
            if part:  # skip trailing empty string
                result.append(part)
    return result


def _recursive_split(text: str, separators: list[str]) -> list[str]:
    """Recursively split text into pieces that are each <= CHUNK_SIZE tokens.

    Tries the first separator in the list; if any resulting piece is still
    too large, recurses with the next separator.
    """
    # Base case: text already fits
    if _token_count(text) <= CHUNK_SIZE:
        return [text]

    # If no separators left, hard-split by tokens
    if not separators:
        return _hard_split_by_tokens(text)

    current_sep = separators[0]
    remaining_seps = separators[1:]

    pieces = _split_by_separator(text, current_sep)

    # If separator didn't actually split anything, try the next one
    if len(pieces) <= 1:
        return _recursive_split(text, remaining_seps)

    # Merge small consecutive pieces, then recurse on any that are still too large
    chunks = []
    for piece in pieces:
        if _token_count(piece) <= CHUNK_SIZE:
            chunks.append(piece)
        else:
            chunks.extend(_recursive_split(piece, remaining_seps))

    return chunks


def _hard_split_by_tokens(text: str, max_tokens: int = CHUNK_SIZE) -> list[str]:
    """Absolute last resort: split by token count directly."""
    tokens = _encoder.encode(text)
    pieces = []
    for i in range(0, len(tokens), max_tokens):
        piece = _encoder.decode(tokens[i : i + max_tokens])
        pieces.append(piece)
    return pieces


def _merge_with_overlap(pieces: list[str]) -> list[str]:
    """Merge split pieces into chunks with token overlap."""
    if not pieces:
        return []

    chunks = []
    current_text = pieces[0]

    for i in range(1, len(pieces)):
        combined = current_text + pieces[i]

        if _token_count(combined) <= CHUNK_SIZE:
            # Merge into current chunk
            current_text = combined
        else:
            # Finalize current chunk
            chunks.append(current_text)

            # Create overlap: take the last CHUNK_OVERLAP tokens of the current chunk
            current_tokens = _encoder.encode(current_text)
            if len(current_tokens) > CHUNK_OVERLAP:
                overlap_text = _encoder.decode(current_tokens[-CHUNK_OVERLAP:])
            else:
                overlap_text = current_text

            # Start new chunk with overlap + next piece
            current_text = overlap_text + pieces[i]

            # If overlap + piece exceeds limit, drop the overlap
            if _token_count(current_text) > CHUNK_SIZE:
                current_text = pieces[i]

    # Don't forget the last chunk
    if current_text:
        chunks.append(current_text)

    return chunks


def chunk_document(
    text: str, doc_id: str, filename: str, file_type: str
) -> list[dict]:
    """Split text into overlapping chunks and return metadata-enriched dicts.

    Args:
        text: Clean extracted text.
        doc_id: Unique document identifier.
        filename: Original filename.
        file_type: File extension ("pdf", "docx").

    Returns:
        List of chunk dictionaries ready for Qdrant insertion.

    Raises:
        ValueError: If input text is empty.
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    timestamp = datetime.now(timezone.utc).isoformat()

    # Step 1: Recursively split into small pieces
    pieces = _recursive_split(text, SEPARATORS)

    # Step 2: Merge pieces with overlap
    raw_chunks = _merge_with_overlap(pieces)

    # Step 3: Filter out chunks that are too small
    filtered = [c for c in raw_chunks if _token_count(c.strip()) >= MIN_CHUNK_TOKENS]

    total_chunks = len(filtered)

    # Step 4: Build output dicts
    result = []
    for idx, chunk_text in enumerate(filtered):
        result.append(
            {
                "doc_id": doc_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "filename": filename,
                "file_type": file_type,
                "uploaded_at": timestamp,
                "text": chunk_text.strip(),
            }
        )

    return result
