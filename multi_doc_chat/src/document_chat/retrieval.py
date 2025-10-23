import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exception.exception import DocumentPortalException
from multi_doc_chat.logger.logger import CustomLogger
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from pydantic import ValidationError

logger = CustomLogger().get_logger(__name__)

class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Handles:
        1. LLM loading
        2. FAISS retriever initialization
        3. Question rewriting
        4. Context-based retrieval and QA

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5)
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        """
        Initialize ConversationalRAG with optional session_id and optional preloaded retriever.

        Args:
            session_id: Unique session identifier
            retriever: Optional pre-built retriever
        """
        try:
            self.session_id = session_id

            # Load LLM once
            self.llm = self._load_llm()

            # Load prompts from registry
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy initialization of retriever and chain
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            logger.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            logger.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.

        Args:
            index_path: Path to FAISS index directory
            k: Number of documents to return
            index_name: Name of the index file
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            fetch_k: Number of documents to fetch before MMR re-ranking (only for MMR)
            lambda_mult: Diversity parameter for MMR (0=max diversity, 1=max relevance)
            search_kwargs: Custom search kwargs (overrides other parameters if provided)
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            # Load embeddings from ModelLoader
            embeddings = ModelLoader().load_embeddings()

            # Load local FAISS vectorstore
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # Safe if trusted
            )

            # Setup search parameters
            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs["fetch_k"] = fetch_k
                    search_kwargs["lambda_mult"] = lambda_mult

            # Build retriever and LCEL chain
            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()

            logger.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                search_type=search_type,
                k=k,
                fetch_k=fetch_k if search_type == "mmr" else None,
                lambda_mult=lambda_mult if search_type == "mmr" else None,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            logger.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """
        Invoke the LCEL RAG pipeline to generate an answer.

        Args:
            user_input: User's query string
            chat_history: Optional chat history to provide context

        Returns:
            str: Generated answer

        Raises:
            DocumentPortalException: If chain not initialized or answer invalid
        """
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)

            if not answer:
                logger.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."

            # Validate answer type and length using Pydantic
            try:
                validated = ChatAnswer(answer=str(answer))
                answer = validated.answer
            except ValidationError as ve:
                logger.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid chat answer", sys)

            logger.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            logger.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    # ---------- Internal Methods ----------

    def _load_llm(self):
        """Load the configured LLM using ModelLoader."""
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            logger.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            logger.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        """Convert list of documents to single string."""
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        """
        Build the LCEL graph/chain with:
            1. Question rewriting
            2. Document retrieval
            3. Contextual QA generation
        """
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # Retrieve documents for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # Build final chain: use retrieved docs + input + history to generate answer
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            logger.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            logger.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
