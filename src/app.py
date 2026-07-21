import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from src.config import settings
from src.services.minio_service import MinioService
from src.services.vector_store import VectorStoreService
from src.llms.llm_factory import LLMFactory

from fastapi import Depends
from sqlalchemy.orm import Session
from src.database import Base, engine, get_db
from src.models.user import User
from src.models.chat import ChatSession, ChatMessage
from src.models.document import Document, ParentChunk, DocumentJob
from langchain_core.documents import Document as LangchainDocument
from src.services.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token,
    get_current_user,
    get_current_admin
)
from datetime import datetime

# Setup LangChain / LangSmith Tracing
if settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGSMITH_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT_NAME

app = FastAPI(
    title="AuditChatBot API",
    description="A FastAPI service for auditing documents using MinIO, HuggingFace embeddings, FAISS, and LangChain RAG.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instantiations
minio_service = MinioService()
vector_store_service = VectorStoreService()

_ranker = None
def get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        _ranker = Ranker()
    return _ranker

def process_document_background(job_id: int, file_bytes: bytes, filename: str, user_id: int):
    from src.database import SessionLocal
    db = SessionLocal()
    try:
        job = db.query(DocumentJob).filter(DocumentJob.id == job_id).first()
        if job:
            job.status = "PROCESSING"
            db.commit()

        # Index document using Parent-Document splitting to Qdrant + DB
        vector_store_service.process_and_index_document(
            file_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
            db=db
        )

        if job:
            job.status = "COMPLETED"
            db.commit()
    except Exception as e:
        db.rollback()
        job = db.query(DocumentJob).filter(DocumentJob.id == job_id).first()
        if job:
            job.status = "FAILED"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()

# Request Pydantic models
class ChatRequest(BaseModel):
    question: str = Field(description="The user query or question about the audited documents.")
    provider: str = Field(default="groq", description="LLM provider: 'groq' or 'openai'")
    model: Optional[str] = Field(default=None, description="Optional custom model name")
    k: int = Field(default=4, description="Number of context documents to retrieve")
    session_id: Optional[int] = Field(default=None, description="Active chat session ID")

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: int

# Chat Memory Pydantic Schemas
class ChatSessionResponse(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    sources: Optional[List[dict]] = None
    created_at: datetime

    class Config:
        from_attributes = True

# User/Auth Pydantic Schemas
class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponseSchema(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class UserUpdateSchema(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None

class AdminUserCreateSchema(BaseModel):
    email: str
    password: str
    role: str = "user"

# Startup event to create tables and seed default admin
@app.on_event("startup")
def startup_db_seeding():
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        admin_exists = db.query(User).filter(User.role == "admin").first()
        if not admin_exists:
            hashed_pw = get_password_hash("admin123")
            admin_user = User(
                email="admin@auditing.com",
                hashed_password=hashed_pw,
                role="admin",
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            print("Default admin user created: admin@auditing.com / admin123")
    finally:
        db.close()

# Authentication Endpoints
@app.post("/auth/register", response_model=UserResponseSchema, tags=["Authentication"])
def register_user(user_in: UserRegister, db: Session = Depends(get_db)):
    """
    Registers a new standard user in the system.
    """
    user_exists = db.query(User).filter(User.email == user_in.email).first()
    if user_exists:
        raise HTTPException(status_code=400, detail="A user with this email already exists")
    
    hashed_pw = get_password_hash(user_in.password)
    new_user = User(
        email=user_in.email,
        hashed_password=hashed_pw,
        role="user",
        is_active=True
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/auth/login", tags=["Authentication"])
def login_user(user_in: UserLogin, db: Session = Depends(get_db)):
    """
    Logs in a user and returns a JWT access token.
    """
    user = db.query(User).filter(User.email == user_in.email).first()
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="User account is deactivated")
        
    access_token = create_access_token(data={"sub": user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "email": user.email
    }

@app.get("/auth/me", response_model=UserResponseSchema, tags=["Authentication"])
def get_user_profile(current_user: User = Depends(get_current_user)):
    """
    Returns the current user profile.
    """
    return current_user

# Chat Sessions & Messages API
@app.post("/chat/sessions", response_model=ChatSessionResponse, tags=["Chat Sessions"])
def create_session(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Creates a new chat session (thread) for the current logged-in user.
    """
    new_sess = ChatSession(user_id=current_user.id, title="New Conversation")
    db.add(new_sess)
    db.commit()
    db.refresh(new_sess)
    return new_sess

@app.get("/chat/sessions", response_model=List[ChatSessionResponse], tags=["Chat Sessions"])
def get_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Retrieves all chat sessions for the current logged-in user.
    """
    sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).all()
    return sessions

@app.delete("/chat/sessions/{session_id}", tags=["Chat Sessions"])
def delete_session(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Deletes a specific chat session and all its messages.
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    db.delete(session)
    db.commit()
    return {"message": "Chat session and all messages successfully deleted."}

@app.get("/chat/sessions/{session_id}/messages", response_model=List[ChatMessageResponse], tags=["Chat Sessions"])
def get_session_messages(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Retrieves all messages for a specific chat session.
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session.messages

# Admin Management Endpoints
@app.get("/admin/users", response_model=List[UserResponseSchema], tags=["Admin User Management"])
def list_users(db: Session = Depends(get_db), current_admin: User = Depends(get_current_admin)):
    """
    Lists all users in the system. (Admin only)
    """
    return db.query(User).order_by(User.id).all()

@app.post("/admin/users", response_model=UserResponseSchema, tags=["Admin User Management"])
def admin_create_user(user_in: AdminUserCreateSchema, db: Session = Depends(get_db), current_admin: User = Depends(get_current_admin)):
    """
    Creates a new user with a specific role. (Admin only)
    """
    user_exists = db.query(User).filter(User.email == user_in.email).first()
    if user_exists:
        raise HTTPException(status_code=400, detail="A user with this email already exists")
        
    hashed_pw = get_password_hash(user_in.password)
    new_user = User(
        email=user_in.email,
        hashed_password=hashed_pw,
        role=user_in.role,
        is_active=True
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.put("/admin/users/{user_id}", response_model=UserResponseSchema, tags=["Admin User Management"])
def admin_update_user(user_id: int, user_in: UserUpdateSchema, db: Session = Depends(get_db), current_admin: User = Depends(get_current_admin)):
    """
    Updates user role or active status. (Admin only)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user_id == current_admin.id and user_in.is_active is False:
        raise HTTPException(status_code=400, detail="Admins cannot deactivate their own account")
        
    if user_in.role is not None:
        user.role = user_in.role
    if user_in.is_active is not None:
        user.is_active = user_in.is_active
        
    db.commit()
    db.refresh(user)
    return user

@app.delete("/admin/users/{user_id}", tags=["Admin User Management"])
def admin_delete_user(user_id: int, db: Session = Depends(get_db), current_admin: User = Depends(get_current_admin)):
    """
    Deletes a user from the system. (Admin only)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user_id == current_admin.id:
        raise HTTPException(status_code=400, detail="Admins cannot delete their own account")
        
    db.delete(user)
    db.commit()
    return {"message": "User successfully deleted"}

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "AuditChatBotAPI"}

@app.get("/documents", tags=["Documents"])
def get_documents(current_user: User = Depends(get_current_user)):
    """
    Retrieves a list of all documents uploaded and stored in MinIO.
    """
    try:
        docs = minio_service.list_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")

@app.get("/documents/download/{filename}", tags=["Documents"])
def download_document(filename: str, token: Optional[str] = Query(None), db: Session = Depends(get_db)):
    """
    Downloads a document from MinIO. Supports query parameter token.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token is missing")
    
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
        
    email = payload.get("sub")
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Inactive or invalid user")

    try:
        file_bytes = minio_service.download_file(filename)
        
        # Determine content type
        content_type = "application/octet-stream"
        if filename.lower().endswith(".pdf"):
            content_type = "application/pdf"
        elif filename.lower().endswith(".docx"):
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
        return Response(
            content=file_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found or failed to download: {str(e)}")

@app.post("/upload", tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Uploads a PDF or DOCX document to MinIO, and schedules text extraction,
    parent-child chunking, and Qdrant indexing in the background.
    """
    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".pdf") or filename_lower.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    try:
        # Read file bytes
        file_bytes = await file.read()
        file_size = len(file_bytes)

        # 1. Upload to MinIO
        import io
        file_stream = io.BytesIO(file_bytes)
        minio_service.upload_file(
            object_name=file.filename,
            data=file_stream,
            length=file_size,
            content_type=file.content_type
        )

        # 2. Create the DocumentJob in DB
        job = DocumentJob(
            filename=file.filename,
            status="PENDING",
            user_id=current_user.id
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # 3. Schedule background processing
        background_tasks.add_task(
            process_document_background,
            job_id=job.id,
            file_bytes=file_bytes,
            filename=file.filename,
            user_id=current_user.id
        )

        return {
            "message": "File successfully uploaded. Ingestion has been scheduled in the background.",
            "filename": file.filename,
            "size": file_size,
            "job_id": job.id,
            "status": "PENDING"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process and index file: {str(e)}")

@app.post("/documents/sync", tags=["Documents"])
def sync_documents_from_minio(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Downloads PDF/DOCX files stored in MinIO and schedules them for indexing in Qdrant/PostgreSQL
    if they are not already indexed.
    """
    try:
        # 1. Retrieve the list of files from MinIO
        minio_docs = minio_service.list_documents()
        if not minio_docs:
            return {
                "message": "No documents found in MinIO bucket to sync.",
                "jobs_created": 0,
                "synced_documents": []
            }

        # 2. Retrieve already indexed filenames for this user
        indexed_files = vector_store_service.get_indexed_filenames(current_user.id, db)

        synced_files = []
        jobs_created = 0

        # 3. Iterate and process each PDF or DOCX file
        for doc in minio_docs:
            filename = doc["name"]
            filename_lower = filename.lower()
            if not (filename_lower.endswith(".pdf") or filename_lower.endswith(".docx")):
                continue

            # Skip if the file is already indexed
            if filename in indexed_files:
                continue

            # Skip if there's already an active sync/ingestion job for this file
            active_job = db.query(DocumentJob).filter(
                DocumentJob.filename == filename,
                DocumentJob.user_id == current_user.id,
                DocumentJob.status.in_(["PENDING", "PROCESSING"])
            ).first()
            if active_job:
                continue

            # Download file from MinIO
            file_bytes = minio_service.download_file(filename)

            # Create job record
            job = DocumentJob(
                filename=filename,
                status="PENDING",
                user_id=current_user.id
            )
            db.add(job)
            db.commit()
            db.refresh(job)

            # Add to background tasks
            background_tasks.add_task(
                process_document_background,
                job_id=job.id,
                file_bytes=file_bytes,
                filename=filename,
                user_id=current_user.id
            )
            
            synced_files.append({
                "filename": filename,
                "job_id": job.id
            })
            jobs_created += 1

        return {
            "message": f"Successfully scheduled {jobs_created} new document(s) from MinIO for indexing.",
            "jobs_created": jobs_created,
            "synced_documents": synced_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync documents from MinIO: {str(e)}")

@app.get("/documents/jobs/{job_id}", tags=["Documents"])
def get_job_status(job_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Retrieves the status of a background document ingestion job.
    """
    job = db.query(DocumentJob).filter(
        DocumentJob.id == job_id,
        DocumentJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
        
    return {
        "job_id": job.id,
        "filename": job.filename,
        "status": job.status,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_with_docs(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Retrieves context for the user question from FAISS (using memory query condensation if history exists),
    queries the LLM incorporating conversation history, saves the messages, and returns the response.
    """
    try:
        # 1. Fetch or create ChatSession
        if request.session_id is not None:
            session = db.query(ChatSession).filter(ChatSession.id == request.session_id, ChatSession.user_id == current_user.id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")
        else:
            session = ChatSession(user_id=current_user.id, title="New Conversation")
            db.add(session)
            db.commit()
            db.refresh(session)

        # 2. Get last 8 messages of context history
        history_messages = session.messages
        context_history = history_messages[-8:] if len(history_messages) > 8 else history_messages

        # 3. Get LLM instance
        llm = LLMFactory.get_llm(provider=request.provider, model_name=request.model)

        # 4. Condense query if history exists
        search_query = request.question
        if context_history:
            history_str = ""
            for msg in context_history:
                role_label = "User" if msg.role == "user" else "Assistant"
                history_str += f"{role_label}: {msg.content}\n"
            
            condense_prompt = (
                "Given the following chat history and a follow-up question, rewrite the follow-up question "
                "to be a standalone question that contains all necessary context from the history. "
                "Do NOT answer the question, just rewrite it. Output ONLY the standalone question text, do not add any comments or notes.\n\n"
                f"Chat History:\n{history_str}\n"
                f"Follow-up Question: {request.question}\n\n"
                "Standalone Question:"
            )
            from langchain_core.messages import HumanMessage
            rewrite_res = llm.invoke([HumanMessage(content=condense_prompt)])
            standalone_res = rewrite_res.content.strip()
            if standalone_res:
                search_query = standalone_res

        # 5. Similarity search with (possibly condensed) query
        retrieved_docs = vector_store_service.similarity_search(
            query=search_query,
            user_id=current_user.id,
            db=db,
            k=15
        )
        
        # 6. Rerank using FlashRank
        docs = []
        if retrieved_docs:
            try:
                ranker = get_ranker()
                passages = [
                    {
                        "id": i,
                        "text": doc.page_content,
                        "meta": doc.metadata
                    }
                    for i, doc in enumerate(retrieved_docs)
                ]
                from flashrank import RerankRequest
                rerank_request = RerankRequest(query=search_query, passages=passages)
                results = ranker.rerank(rerank_request)
                
                # Filter to top request.k (default 4) reranked chunks
                top_results = results[:request.k]
                docs = [
                    LangchainDocument(
                        page_content=res["text"],
                        metadata=res["meta"]
                    )
                    for res in top_results
                ]
            except Exception as re_err:
                print(f"Warning: Reranking failed: {re_err}. Falling back to default top {request.k} results.")
                docs = retrieved_docs[:request.k]
        
        # 7. Prepare context string
        context_parts = []
        if docs:
            for i, doc in enumerate(docs):
                source_info = f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', 0) + 1}]"
                context_parts.append(f"--- Document Chunk {i+1} {source_info} ---\n{doc.page_content}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "No relevant document context found."

        # 7. Create system prompt
        system_prompt = (
            "You are a professional auditor and compliance specialist. Use the following retrieved "
            "context chunks to answer the question. If you do not know the answer based on the context, "
            "say that you cannot find the answer in the uploaded documents. Do not make up answers.\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            "CRITICAL INSTRUCTION:\n"
            "At the very end of your response, in a new line, you MUST append a source reference block specifying "
            "which document names you actually used to answer the question. Format it exactly like this:\n"
            "Sources: [filename1.pdf, filename2.docx]\n"
            "If you did not find the answer in the context, or if the user query is a general greeting or non-compliance "
            "question (like hello, hi, thank you, etc.) and you did not use the documents to answer it, write exactly:\n"
            "Sources: [None]"
        )
        
        # 8. Compile full message thread list (System Context + History + Current Question)
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        langchain_messages = [SystemMessage(content=system_prompt)]
        
        for msg in context_history:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            else:
                langchain_messages.append(AIMessage(content=msg.content))
                
        langchain_messages.append(HumanMessage(content=request.question))
        
        # 9. Get response from LLM
        response = llm.invoke(langchain_messages)
        answer_text = response.content.strip()
        
        # 10. Parse sources
        import re
        sources_match = re.search(r"Sources:\s*\[(.*?)\]", answer_text, re.IGNORECASE)
        
        used_sources = []
        if sources_match:
            source_content = sources_match.group(1).strip()
            answer_text = re.sub(r"\n*Sources:\s*\[.*?\]", "", answer_text, flags=re.IGNORECASE).strip()
            
            if source_content.lower() != "none" and source_content != "":
                used_sources = [s.strip() for s in source_content.split(",")]
        else:
            # Fallback heuristic: if the LLM successfully answered the question using context
            # but omitted the "Sources: [...]" instruction format, attribute it to retrieved docs.
            lowered_ans = answer_text.lower()
            not_found_flags = [
                "cannot find the answer",
                "do not know",
                "no relevant document",
                "not mentioned in the provided",
                "not found in the uploaded"
            ]
            if not any(flag in lowered_ans for flag in not_found_flags) and docs:
                seen_srcs = set()
                for doc in docs:
                    src_name = doc.metadata.get("source")
                    if src_name and src_name not in seen_srcs:
                        seen_srcs.add(src_name)
                        used_sources.append(src_name)

        # 11. Format source objects
        filtered_sources = []
        if used_sources and docs:
            for doc in docs:
                source_name = doc.metadata.get("source")
                if source_name in used_sources:
                    if not any(x["source"] == source_name for x in filtered_sources):
                        filtered_sources.append({
                            "source": source_name,
                            "page": doc.metadata.get("page", 0) + 1,
                            "snippet": doc.page_content[:150] + "..."
                        })

        # 12. Update chat session title dynamically if this was the first query
        if len(history_messages) == 0:
            title_text = request.question[:35]
            if len(request.question) > 35:
                title_text += "..."
            session.title = title_text
            db.add(session)

        # 13. Save queries to database
        user_msg = ChatMessage(session_id=session.id, role="user", content=request.question)
        db.add(user_msg)
        
        assistant_msg = ChatMessage(session_id=session.id, role="assistant", content=answer_text, sources=filtered_sources)
        db.add(assistant_msg)
        
        db.commit()

        return ChatResponse(
            answer=answer_text,
            sources=filtered_sources,
            session_id=session.id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
