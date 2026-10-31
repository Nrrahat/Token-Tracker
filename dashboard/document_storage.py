"""Confined, bounded storage for user-uploaded reader documents."""

from pathlib import Path


ALLOWED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".csv", ".json"}
MAX_DOCUMENT_BYTES = 1_000_000


class DocumentUploadError(ValueError):
    """Raised when an uploaded document violates storage policy."""


def store_document(root: str | Path, filename: str, content: bytes) -> Path:
    directory = Path(root).expanduser().resolve()
    name = Path(filename).name
    if not name or name in {".", ".."} or name != filename:
        raise DocumentUploadError("Only a plain file name is allowed")
    if Path(name).suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise DocumentUploadError("File type is not allowed")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentUploadError("Document exceeds the 1 MB upload limit")

    directory.mkdir(parents=True, exist_ok=True)
    destination = (directory / name).resolve()
    try:
        destination.relative_to(directory)
    except ValueError as error:
        raise DocumentUploadError("Document path escapes the configured storage directory") from error
    try:
        with destination.open("xb") as document_file:
            document_file.write(content)
    except FileExistsError as error:
        raise DocumentUploadError("A document with this name already exists") from error
    return destination