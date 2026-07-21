import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HF_TOKEN = os.getenv("HF_TOCKEN")  # Note the typo in .env is handled

    # LangChain / LangSmith Settings
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT_NAME", "AuditChatBotAPI")

    # MinIO Settings
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "auditchatbot")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in ("true", "1", "yes")

    # Vector store paths
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "auditchatbot")

    # PostgreSQL Database Settings
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/auditdb")

    # JWT Authentication Settings
    JWT_SECRET = os.getenv("JWT_SECRET", "5d4d3a246a4891b2cde3f8e75b6a7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

settings = Settings()
