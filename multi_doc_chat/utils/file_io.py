from __future__ import annotations
import re
import uuid
from pathlib import Path
from typing import Iterable, List
from multi_doc_chat.logger.logger import CustomLogger
from multi_doc_chat.exception.exception import DocumentPortalException

# Supported file types for ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"}

# Initialize local logger instance
log = CustomLogger().get_logger(__name__)


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """
    Save uploaded files (e.g., from Streamlit) to a target directory and return saved file paths.

    Handles Starlette UploadFile objects (with .filename and .file) and generic objects
    exposing .name or .read(). Unsupported file types are skipped with a warning.

    Args:
        uploaded_files (Iterable): Collection of uploaded file objects.
        target_dir (Path): Directory to save uploaded files.

    Returns:
        List[Path]: List of saved file paths.

    Raises:
        DocumentPortalException: If saving any file fails.
    """
    try:
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for uf in uploaded_files:
            # Extract filename safely from different object types
            name = getattr(uf, "filename", getattr(uf, "name", "file"))
            ext = Path(name).suffix.lower()

            # Skip unsupported file extensions
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=name)
                continue

            # Sanitize file name (alphanumeric, dash, underscore) and lowercase
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(name).stem).lower()

            # Generate unique file name to avoid collisions
            fname = f"{safe_name}_{uuid.uuid4().hex[:6]}{ext}"
            # Optional alternative unique naming (currently overwrites previous line)
            fname = f"{uuid.uuid4().hex[:8]}{ext}"

            out = target_dir / fname

            # Write file content to disk
            with open(out, "wb") as f:
                if hasattr(uf, "file") and hasattr(uf.file, "read"):
                    # Starlette UploadFile.file
                    f.write(uf.file.read())
                elif hasattr(uf, "read"):
                    # Generic file-like object
                    data = uf.read()
                    if isinstance(data, memoryview):
                        data = data.tobytes()
                    f.write(data)
                else:
                    # Fallback for objects with getbuffer()
                    buf = getattr(uf, "getbuffer", None)
                    if callable(buf):
                        data = buf()
                        if isinstance(data, memoryview):
                            data = data.tobytes()
                        f.write(data)
                    else:
                        # Object does not have readable interface
                        raise ValueError("Unsupported uploaded file object; no readable interface")

            # Keep track of saved file path
            saved.append(out)
            log.info("File saved for ingestion", uploaded=name, saved_as=str(out))

        return saved

    except Exception as e:
        # Log and raise custom exception if any file fails to save
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e
