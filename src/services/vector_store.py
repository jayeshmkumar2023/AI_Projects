import os
import tempfile
from typing import List
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain_community.document_loaders import PyPDFLoader

from src.config import settings
from src.models.document import Document as DB_Document, ParentChunk

class VectorStoreService:
    def __init__(self):
        # Set token in environment if it exists
        if settings.HF_TOKEN:
            os.environ["HF_TOKEN"] = settings.HF_TOKEN
        
        # Load HuggingFace embeddings model (matches all-miniLM-L6-v2)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-miniLM-L6-v2")
        
        # Initialize Qdrant Client
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self._ensure_collection_exists()

        # Initialize LangChain Qdrant Vector Store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=self.embeddings
        )

        # Set up Parent and Child text splitters
        # Parent chunks (larger context to send to LLM)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        # Child chunks (smaller chunks optimized for vector search matching)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )

    def _ensure_collection_exists(self):
        """
        Creates the Qdrant collection if it does not exist.
        """
        try:
            collections = [c.name for c in self.qdrant_client.get_collections().collections]
            if settings.QDRANT_COLLECTION not in collections:
                # all-miniLM-L6-v2 has 384 dimensions
                self.qdrant_client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=qdrant_models.VectorParams(
                        size=384,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                print(f"Qdrant collection '{settings.QDRANT_COLLECTION}' created successfully.")
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant or create collection: {e}")

    def process_and_index_document(self, file_bytes: bytes, filename: str, user_id: int, db: Session) -> int:
        """
        Parses a document, splits it into parent and child chunks, saves parent chunks
        to PostgreSQL, and indexes child chunks in Qdrant.
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
            else:
                return 0
            
            pages = loader.load()

            # Clean up existing records and vectors for the same filename/user to prevent duplicates
            existing_doc = db.query(DB_Document).filter(
                DB_Document.filename == filename,
                DB_Document.user_id == user_id
            ).first()
            
            if existing_doc:
                # Remove vectors from Qdrant associated with this filename and user
                try:
                    self.qdrant_client.delete(
                        collection_name=settings.QDRANT_COLLECTION,
                        points_selector=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="metadata.source",
                                    match=qdrant_models.MatchValue(value=filename)
                                ),
                                qdrant_models.FieldCondition(
                                    key="metadata.user_id",
                                    match=qdrant_models.MatchValue(value=user_id)
                                )
                            ]
                        )
                    )
                except Exception as e:
                    print(f"Warning: Failed to delete old vectors from Qdrant: {e}")

                # SQLAlchemy cascade deletes parent_chunks automatically
                db.delete(existing_doc)
                db.commit()

            # Create new Document DB record
            doc_record = DB_Document(
                filename=filename,
                bucket_name=settings.MINIO_BUCKET,
                file_size=len(file_bytes),
                user_id=user_id
            )
            db.add(doc_record)
            db.commit()
            db.refresh(doc_record)

            # Generate Parent Chunks
            parent_docs = self.parent_splitter.split_documents(pages)
            if not parent_docs:
                return 0

            parent_records = []
            child_documents = []

            for parent_idx, parent_doc in enumerate(parent_docs):
                parent_id = f"{doc_record.id}_{parent_idx}"
                page_num = parent_doc.metadata.get("page", 0)

                # Create ParentChunk record
                parent_record = ParentChunk(
                    id=parent_id,
                    document_id=doc_record.id,
                    text=parent_doc.page_content,
                    chunk_index=parent_idx,
                    page=page_num
                )
                parent_records.append(parent_record)

                # Split parent chunk into smaller child chunks
                child_texts = self.child_splitter.split_text(parent_doc.page_content)
                for child_idx, child_text in enumerate(child_texts):
                    child_doc = LangchainDocument(
                        page_content=child_text,
                        metadata={
                            "parent_id": parent_id,
                            "source": filename,
                            "user_id": user_id,
                            "page": page_num,
                            "child_idx": child_idx
                        }
                    )
                    child_documents.append(child_doc)

            # Bulk save parent chunks
            db.bulk_save_objects(parent_records)
            db.commit()

            # Index child chunks in Qdrant
            if child_documents:
                self.vector_store.add_documents(child_documents)

            return len(child_documents)
        finally:
            # Clean up the temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def similarity_search(self, query: str, user_id: int, db: Session, k: int = 15) -> List[LangchainDocument]:
        """
        Retrieves the top-k small child chunks from Qdrant (filtered by user_id),
        and fetches their corresponding large parent chunks from PostgreSQL.
        """
        try:
            filter_query = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.user_id",
                        match=qdrant_models.MatchValue(value=user_id)
                    )
                ]
            )
            
            # Query child chunks
            child_docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_query
            )

            if not child_docs:
                return []

            # Extract parent IDs while maintaining the order of child relevance
            parent_ids = [doc.metadata.get("parent_id") for doc in child_docs if doc.metadata.get("parent_id")]
            
            # Deduplicate parent IDs while keeping their first occurrence order
            seen = set()
            unique_parent_ids = []
            for pid in parent_ids:
                if pid not in seen:
                    seen.add(pid)
                    unique_parent_ids.append(pid)

            if not unique_parent_ids:
                return []

            # Retrieve Parent Chunks in bulk
            parent_chunks = db.query(ParentChunk).filter(
                ParentChunk.id.in_(unique_parent_ids)
            ).all()

            # Map to easily fetch parent chunk objects by ID
            parent_map = {p.id: p for p in parent_chunks}

            # Reconstruct list of LangchainDocuments in the original search rank order
            retrieved_parents = []
            for pid in unique_parent_ids:
                p_chunk = parent_map.get(pid)
                if p_chunk:
                    retrieved_parents.append(
                        LangchainDocument(
                            page_content=p_chunk.text,
                            metadata={
                                "source": p_chunk.document.filename,
                                "page": p_chunk.page,
                                "parent_id": p_chunk.id
                            }
                        )
                    )
            return retrieved_parents
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []

    def get_indexed_filenames(self, user_id: int, db: Session) -> set:
        """
        Returns a set of all unique filenames currently indexed for the given user.
        """
        try:
            docs = db.query(DB_Document).filter(DB_Document.user_id == user_id).all()
            return {doc.filename for doc in docs}
        except Exception as e:
            print(f"Error fetching indexed filenames: {e}")
            return set()
