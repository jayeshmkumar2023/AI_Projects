from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database import Base
from src.models.user import User

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    bucket_name = Column(String, nullable=False)
    file_size = Column(Integer, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", backref="documents")
    parent_chunks = relationship("ParentChunk", back_populates="document", cascade="all, delete-orphan")

class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    id = Column(String, primary_key=True)  # Format: "filename_chunkidx" or uuid
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="parent_chunks")

class DocumentJob(Base):
    __tablename__ = "document_jobs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    status = Column(String, nullable=False, default="PENDING")  # PENDING, PROCESSING, COMPLETED, FAILED
    error_message = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", backref="document_jobs")
