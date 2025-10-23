from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.exception import DocumentPortalException
from fastapi import UploadFile

# Supported file extensions for document ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Load documents from given file paths using appropriate loader based on file extension.

    Args:
        paths (Iterable[Path]): List of file paths to load.

    Returns:
        List[Document]: List of LangChain Document objects.

    Raises:
        DocumentPortalException: If any document fails to load.
    """
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()

            # Choose loader based on file extension
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                # Skip unsupported file types
                log.warning("Unsupported extension skipped", path=str(p))
                continue

            # Load and append documents
            docs.extend(loader.load())

        log.info("Documents loaded", count=len(docs))
        return docs

    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e
    

class FastAPIFileAdapter:
    """
    Adapter for FastAPI UploadFile objects to provide a simple interface
    with .name and .getbuffer() methods, compatible with generic file handling.
    """

    def __init__(self, uf: UploadFile):
        """
        Initialize adapter with a FastAPI UploadFile.

        Args:
            uf (UploadFile): FastAPI UploadFile instance.
        """
        self._uf = uf
        self.name = uf.filename or "file"  # Fallback name if filename is None

    def getbuffer(self) -> bytes:
        """
        Return the full content of the uploaded file as bytes.

        Returns:
            bytes: File content.
        """
        self._uf.file.seek(0)  # Reset file pointer to start
        return self._uf.file.read()
