import os
import asyncio
from typing import Dict, Any
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from serviices import medical_chatbot
from confiig import settings, is_emergency_symptom


app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events"""
    # Startup
    print("🏥 Starting Clinical Diagnostics Chatbot...")
    print(f"Backend running on port {settings.backend_port}")
    
    app_state["chatbot"] = medical_chatbot
    print("✅ Medical chatbot services initialized")
    
    yield
  
    print("👋 Shutting down Clinical Diagnostics Chatbot...")


app = FastAPI(
    title="Clinical Diagnostics Chatbot API",
    description="AI-powered medical diagnostics with RAG and conversation management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],  # Streamlit + React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple request/response models (no Pydantic headaches)
class ChatRequest:
    def __init__(self, data: dict):
        self.message = data.get("message", "")
        self.session_id = data.get("session_id", "default")
        
    def validate(self):
        if not self.message.strip():
            raise ValueError("Message cannot be empty")
        if len(self.message) > 2000:
            raise ValueError("Message too long (max 2000 characters)")


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "Clinical Diagnostics Chatbot API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/chat": "POST - Main chat endpoint",
            "/upload": "POST - Document upload",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.post("/chat")
async def chat_endpoint(request_data: dict):
    """
    Main chat endpoint - processes medical queries
    Based on search results pattern for chatbot endpoints
    """
    try:
     
        chat_request = ChatRequest(request_data)
        chat_request.validate()
        
        # Process with medical chatbot service
        result = await medical_chatbot.process_medical_query(
            message=chat_request.message,
            session_id=chat_request.session_id
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
        
        
        return {
            "success": True,
            "response": result["response"],
            "urgency_level": result["urgency_level"],
            "emergency": result.get("emergency", False),
            "possible_conditions": result.get("possible_conditions", []),
            "session_id": result["session_id"],
            "timestamp": datetime.now().isoformat(),
            "disclaimer": result["disclaimer"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Document upload endpoint
    Based on search results showing file upload patterns
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.allowed_file_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {settings.allowed_file_types}"
            )
        
        content = await file.read()
        if len(content) > settings.max_upload_size:
            raise HTTPException(status_code=400, detail="File too large")
        
    
        os.makedirs(settings.upload_dir, exist_ok=True)
        file_path = os.path.join(settings.upload_dir, f"{session_id}_{file.filename}")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
       
        result = await medical_chatbot.doc_service.process_document(file_path, file_extension)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "file_size": len(content),
            "extracted_info": result["medical_info"],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail="File processing failed")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get conversation history for a session"""
    try:
        context = medical_chatbot.context_service.get_context(session_id)
        
        if not context:
            return {
                "success": False,
                "message": "Session not found"
            }
        
        return {
            "success": True,
            "session_id": session_id,
            "created_at": context["created_at"].isoformat(),
            "message_count": len(context["messages"]),
            "summary": medical_chatbot.context_service.get_conversation_summary(session_id),
            "medical_context": context["medical_context"]
        }
        
    except Exception as e:
        print(f"❌ Session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Based on search results showing health check patterns
    """
    try:
        services_status = {}
        
        if settings.groq_api_key:
            services_status["groq"] = "configured"
        else:
            services_status["groq"] = "not_configured"
        
        if settings.huggingface_api_token:
            services_status["huggingface"] = "configured"
        else:
            services_status["huggingface"] = "not_configured"
        
        # Check file directories
        services_status["upload_dir"] = "ready" if os.path.exists(settings.upload_dir) else "missing"
        services_status["docs_dir"] = "ready" if os.path.exists(settings.documents_dir) else "missing"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.app_version,
            "services": services_status,
            "uptime": "running"
        }
        
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/emergency-check")
async def emergency_check(symptoms: str):
    """Quick emergency symptom check"""
    try:
        is_emergency = is_emergency_symptom(symptoms)
        
        return {
            "symptoms": symptoms,
            "is_emergency": is_emergency,
            "message": settings.emergency_message if is_emergency else "No immediate emergency detected",
            "urgency_level": "emergency" if is_emergency else "normal"
        }
        
    except Exception as e:
        print(f"❌ Emergency check error: {e}")
        raise HTTPException(status_code=500, detail="Emergency check failed")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "Check available endpoints at /"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Please try again later"}
    )

if __name__ == "__main__":
    print("🚀 Starting Clinical Diagnostics Chatbot API...")
    print(f"🌐 API will be available at: http://localhost:{settings.backend_port}")
    print(f"📚 API documentation at: http://localhost:{settings.backend_port}/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=settings.debug,
        log_level="info"
    )