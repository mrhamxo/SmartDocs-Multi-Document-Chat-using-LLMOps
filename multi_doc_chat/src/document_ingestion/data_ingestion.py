from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.exception import DocumentPortalException
import json
import uuid
from datetime import datetime
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.document_ops import load_documents
import hashlib
import sys

def generate_session_id() -> str:
    """
    Generate a unique session ID with timestamp and random UUID.

    Returns:
        str: Unique session identifier, e.g., 'session_20251020_173205_a1b2c3d4'.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


class ChatIngestor:
    """
    Handles document ingestion, splitting, FAISS index creation, and retriever setup.

    Attributes:
        model_loader (ModelLoader): Loads embeddings and LLMs.
        use_session (bool): Whether to create session-specific directories.
        session_id (str): Session identifier.
        temp_base (Path): Base directory for temporary files.
        faiss_base (Path): Base directory for FAISS indices.
        temp_dir (Path): Actual temp directory resolved for session.
        faiss_dir (Path): Actual FAISS directory resolved for session.
    """

    def __init__(self,
                 temp_base: str = "data",
                 faiss_base: str = "faiss_index",
                 use_session_dirs: bool = True,
                 session_id: Optional[str] = None):
        """
        Initialize ChatIngestor with directories and session info.

        Args:
            temp_base (str): Base path for temporary files.
            faiss_base (str): Base path for FAISS indices.
            use_session_dirs (bool): Whether to create session-specific subdirectories.
            session_id (Optional[str]): Optional session ID; generated if None.

        Raises:
            DocumentPortalException: If initialization fails.
        """
        try:
            # Load model loader for embeddings and LLMs
            self.model_loader = ModelLoader()

            # Session handling
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            # Create base directories if not exist
            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            # Resolve directories for session
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            # Log initialization
            log.info("ChatIngestor initialized",
                     session_id=self.session_id,
                     temp_dir=str(self.temp_dir),
                     faiss_dir=str(self.faiss_dir),
                     sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e

    def _resolve_dir(self, base: Path) -> Path:
        """
        Resolve the directory for a session or return base directory.

        Args:
            base (Path): Base path to resolve.

        Returns:
            Path: Resolved path for storing data.
        """
        if self.use_session:
            d = base / self.session_id  # e.g., "faiss_index/session_id"
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base  # fallback to base directory

    def _split(self, docs: List['Document'], chunk_size=1000, chunk_overlap=200) -> List['Document']:
        """
        Split documents into chunks for embedding generation.

        Args:
            docs (List[Document]): List of documents to split.
            chunk_size (int): Number of characters per chunk.
            chunk_overlap (int): Overlap between consecutive chunks.

        Returns:
            List[Document]: List of split document chunks.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks

    def built_retriver(self,
                       uploaded_files: Iterable,
                       *,
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200,
                       k: int = 5,
                       search_type: str = "mmr",
                       fetch_k: int = 20,
                       lambda_mult: float = 0.5):
        """
        Build a FAISS retriever from uploaded files.

        Args:
            uploaded_files (Iterable): Uploaded document files.
            chunk_size (int): Chunk size for splitting.
            chunk_overlap (int): Overlap size for splitting.
            k (int): Number of results to return.
            search_type (str): Search strategy (e.g., 'mmr').
            fetch_k (int): Number of documents to fetch for MMR.
            lambda_mult (float): Diversity parameter for MMR.

        Returns:
            Retriever: Configured retriever object.

        Raises:
            DocumentPortalException: If retriever building fails.
        """
        try:
            # Save uploaded files to temp directory
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            # Split documents into chunks
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Initialize FAISS manager
            fm = FaissManager(self.faiss_dir, self.model_loader)

            # Extract content and metadata
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]

            # Load existing index or create new
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)

            # Add new documents to FAISS index
            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            # Configure search parameters
            search_kwargs = {"k": k}
            if search_type == "mmr":
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult
                log.info("Using MMR search", k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

            return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}  # Allowed file types


class FaissManager:
    """
    Manages FAISS vector index creation, loading, updating, and metadata tracking.

    Attributes:
        index_dir (Path): Directory to store FAISS index files.
        meta_path (Path): Path to JSON metadata file.
        _meta (Dict): Tracks ingested chunks to avoid duplicates.
        model_loader (ModelLoader): Provides embeddings.
        emb: Embedding model.
        vs (Optional[FAISS]): FAISS vector store instance.
    """

    def __init__(self, index_dir: Path, model_loader: Optional['ModelLoader'] = None):
        """
        Initialize FAISS manager.

        Args:
            index_dir (Path): Directory to store index files.
            model_loader (Optional[ModelLoader]): Optional model loader for embeddings.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}  # dict to track ingested rows

        # Load existing metadata if available
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional['FAISS'] = None

    def _exists(self) -> bool:
        """
        Check if FAISS index already exists.

        Returns:
            bool: True if index files exist.
        """
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        """
        Generate a unique fingerprint for a document chunk.

        Args:
            text (str): Document text.
            md (Dict[str, Any]): Metadata.

        Returns:
            str: Fingerprint string.
        """
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        """Save metadata to JSON file."""
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_documents(self, docs: List['Document']):
        """
        Add new document chunks to FAISS index if not already present.

        Args:
            docs (List[Document]): List of document chunks.

        Returns:
            int: Number of new documents added.
        """
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")

        new_docs: List['Document'] = []

        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None):
        """
        Load existing FAISS index or create a new one.

        Args:
            texts (Optional[List[str]]): Document texts for new index.
            metadatas (Optional[List[dict]]): Metadata for new index.

        Returns:
            FAISS: Loaded or newly created FAISS index.

        Raises:
            DocumentPortalException: If no existing index and no texts provided.
        """
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs

        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)

        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs
