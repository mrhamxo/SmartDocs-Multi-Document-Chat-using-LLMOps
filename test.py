from pathlib import Path
import sys
import os
from dotenv import load_dotenv
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

def test_document_ingestion_and_rag():
    """
    Test full document ingestion -> FAISS indexing -> Conversational RAG workflow.
    Steps:
        1. Load test PDF files
        2. Use ChatIngestor to split documents and build FAISS index
        3. Load ConversationalRAG with retriever
        4. Start an interactive chat loop with multi-turn context
    """
    try:
        # Paths to test documents
        test_files = [
            "data\islr.pdf",
        ]

        uploaded_files = []

        # Open files as binary objects for ingestion
        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File does not exist: {file_path}")

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

        # Initialize ChatIngestor for ingestion and FAISS indexing
        ci = ChatIngestor(
            temp_base="data",
            faiss_base="faiss_index",
            use_session_dirs=True
        )

        # Build retriever using MMR for diverse results
        # chunk_size=200: split documents into 200-character chunks
        # chunk_overlap=20: overlap 20 characters between chunks
        # k=5: return top 5 documents per query
        # fetch_k=20: number of docs fetched before MMR re-ranking
        # lambda_mult=0.5: balance between relevance and diversity
        retriever = ci.built_retriver(
            uploaded_files, 
            chunk_size=200, 
            chunk_overlap=20, 
            k=5,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Optional: Use simple similarity search
        # retriever = ci.built_retriver(uploaded_files, chunk_size=200, chunk_overlap=20, k=5, search_type="similarity")

        # Close file handles after ingestion
        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        # Get session ID to locate FAISS index
        session_id = ci.session_id
        index_dir = os.path.join("faiss_index", session_id)

        # Load Conversational RAG with the FAISS retriever
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir, 
            k=5, 
            index_name=os.getenv("FAISS_INDEX_NAME", "index"),
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Interactive multi-turn chat loop
        chat_history = []
        print("\nType 'exit' to quit the chat.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                print("Goodbye!")
                break

            # Generate answer using RAG
            answer = rag.invoke(user_input, chat_history=chat_history)
            print("Assistant:", answer)

            # Maintain conversation history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    test_document_ingestion_and_rag()
