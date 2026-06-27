import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import settings

class VectorStoreService:
    def __init__(self):
        # Set token in environment if it exists
        if settings.HF_TOKEN:
            os.environ["HF_TOKEN"] = settings.HF_TOKEN
        
        # Load HuggingFace embeddings model (matches all-miniLM-L6-v2 from notebook)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-miniLM-L6-v2")
        self.index_dir = settings.FAISS_INDEX_DIR
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def process_and_index_pdf(self, file_bytes: bytes, filename: str) -> int:
        """
        Saves PDF bytes to temporary file, parses it, chunks it,
        and adds it to the FAISS vector database.
        (Maintained for backward compatibility, delegates to process_and_index_document)
        """
        return self.process_and_index_document(file_bytes, filename)

    def process_and_index_document(self, file_bytes: bytes, filename: str) -> int:
        """
        Saves PDF or DOCX file bytes to a temporary file, parses it,
        chunks it, and adds it to the FAISS vector database.
        """
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in (".pdf", ".docx"):
            print(f"Skipping unsupported file format: {suffix} for file {filename}")
            return 0

        # Save bytes to a temp file with the correct suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        try:
            # Load file based on type
            if suffix == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif suffix == ".docx":
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(temp_file_path)
            
            pages = loader.load()
            
            # Inject filename metadata into pages so we can trace source
            for page in pages:
                page.metadata["source"] = filename

            chunks = self.text_splitter.split_documents(pages)
            if not chunks:
                return 0

            # Add to FAISS index (load existing or create new)
            self._add_to_index(chunks)
            return len(chunks)
        finally:
            # Clean up the temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _add_to_index(self, chunks: List[Document]):
        """
        Adds chunks to the FAISS index. Updates if index already exists,
        otherwise creates a new one.
        """
        os.makedirs(os.path.dirname(self.index_dir), exist_ok=True)
        
        # Check if FAISS index files exist
        index_file = os.path.join(self.index_dir, "index.faiss")
        if os.path.exists(index_file):
            # Load existing FAISS index
            db = FAISS.load_local(
                folder_path=self.index_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            # Add new chunks
            db.add_documents(chunks)
        else:
            # Create new FAISS index
            db = FAISS.from_documents(chunks, self.embeddings)

        # Persist index
        db.save_local(self.index_dir)

    def get_retriever(self, search_kwargs: dict = None):
        """
        Returns a retriever if index exists, otherwise None.
        """
        index_file = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_file):
            return None
        
        db = FAISS.load_local(
            folder_path=self.index_dir,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        kwargs = search_kwargs or {"k": 4}
        return db.as_retriever(search_kwargs=kwargs)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs vector search.
        """
        index_file = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_file):
            return []
        
        db = FAISS.load_local(
            folder_path=self.index_dir,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        return db.similarity_search(query, k=k)

    def get_indexed_filenames(self) -> set:
        """
        Returns a set of all unique filenames currently indexed in the FAISS vector database.
        """
        index_file = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_file):
            return set()
        
        try:
            db = FAISS.load_local(
                folder_path=self.index_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            indexed_sources = set()
            for doc in db.docstore._dict.values():
                source = doc.metadata.get("source")
                if source:
                    indexed_sources.add(source)
            return indexed_sources
        except Exception as e:
            print(f"Warning: Error loading FAISS database to check indexed files: {e}")
            return set()
