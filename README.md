# AuditChatBotAPI

AuditChatBotAPI is a robust, modular FastAPI application designed for analyzing and querying audit reports and compliance documentation. It utilizes **MinIO** for document storage, **HuggingFace Embeddings** for semantic vectorization, **FAISS** for vector index caching, and **Groq/OpenAI** for Retrieval-Augmented Generation (RAG) chat answers.

## Architecture

* **FastAPI Router (`src/app.py`)**: Defines endpoints for uploading documents, listing files, querying the model, and checking service health.
* **MinIO Service (`src/services/minio_service.py`)**: Handles object storage (uploading, listing, retrieving documents).
* **Vector Store Service (`src/services/vector_store.py`)**: Extracts PDF text, chunks it, embeds chunks with `all-miniLM-L6-v2`, and updates the persisted FAISS vector index.
* **LLM Factory (`src/llms/llm_factory.py`)**: Dynamically configures and instantiates the chosen LLM connector (Groq or OpenAI).

---

## Configuration

The application reads configuration parameters from `.env`. Ensure your `.env` contains the required keys:

```env
OPENAI_API_KEY="sk-proj-..."
GROQ_API_KEY="gsk_..."
LANGCHAIN_API_KEY="lsv2_..."
LANGCHAIN_PROJECT_NAME="GenAPIApp"

MINIO_ENDPOINT="localhost:9000"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
MINIO_BUCKET="auditchatbot"
MINIO_SECURE="false"
```

---

## Getting Started

### 1. Ensure MinIO is running

You can start a local MinIO container in the background using Docker:

```bash
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

### 2. Run the FastAPI Server

Start the application using:

```bash
uv run python main.py
```

The server will run on `http://localhost:8000`. You can access the auto-generated Swagger UI documentation at `http://localhost:8000/docs`.

---

## API Documentation & Usage Examples

### 1. Health Check
* **Endpoint**: `GET /health`
* **Curl Example**:
  ```bash
  curl http://localhost:8000/health
  ```

### 2. Upload and Index a PDF
* **Endpoint**: `POST /upload`
* **Form Parameters**: `file` (PDF file upload)
* **Curl Example**:
  ```bash
  curl -X POST http://localhost:8000/upload \
    -F "file=@/path/to/audit_report.pdf"
  ```

### 3. List Stored Documents
* **Endpoint**: `GET /documents`
* **Curl Example**:
  ```bash
  curl http://localhost:8000/documents
  ```

### 4. RAG Query / Chat with Documents
* **Endpoint**: `POST /chat`
* **Request Body**:
  ```json
  {
    "question": "What are the key findings or recommendations in the audit report?",
    "provider": "groq",
    "k": 4
  }
  ```
* **Curl Example**:
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the cash shortage policy?", "provider": "groq"}'
  ```
