from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging
import faiss

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, db_path="/home/models/FAISS_INGEST/vectorstore/db_faiss"):
        self.db_path = db_path
        # Initialize embeddings with CPU explicitly
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = None
        self._load_vector_store()

    def _load_vector_store(self):
        """Load the FAISS vector store from disk"""
        try:
            if os.path.exists(self.db_path):
                logger.info(f"Loading FAISS vector store from {self.db_path}")
                # Force CPU mode for FAISS
                faiss.omp_set_num_threads(4)  # Set number of threads for parallel processing
                
                # Load the vector store with CPU-only mode
                self.vector_store = FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS vector store loaded successfully in CPU mode")
            else:
                error_msg = f"Vector database not found at {self.db_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 5):
        """Perform similarity search on the vector database"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            logger.info(f"Performing similarity search for query: {query}")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} relevant documents")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    def get_relevant_documents(self, query: str, k: int = 5):
        """Get relevant documents from the vector database"""
        try:
            results = self.similarity_search(query, k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"Error getting relevant documents: {str(e)}")
            raise 