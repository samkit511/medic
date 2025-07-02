import os
import asyncio
import re
from typing import Dict, Any
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from services import medical_chatbot
from config import settings, is_emergency_symptom
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events"""
    logger.info("🏥 Starting Clinical Diagnostics Chatbot...")
    logger.info(f"Backend running on port {settings.backend_port}")
    
    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info(f"Upload directory ensured: {settings.upload_dir}")
    
    app_state["chatbot"] = medical_chatbot
    logger.info("✅ Medical chatbot services initialized")
    
    yield
    logger.info("👋 Shutting down Clinical Diagnostics Chatbot...")

app = FastAPI(
    title="Clinical Diagnostics Chatbot API",
    description="AI-powered medical diagnostics with RAG and conversation management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    logger.info("Root endpoint accessed")
    return {
        "message": "Clinical Diagnostics Chatbot API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/api/query": "POST - Main chat endpoint",
            "/api/upload": "POST - Document upload",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.post("/api/query")
async def chat_endpoint(request_data: dict):
    """Main chat endpoint - processes medical queries"""
    logger.info(f"Chat endpoint hit with session_id: {request_data.get('session_id', 'default')}")
    try:
        chat_request = ChatRequest(request_data)
        chat_request.validate()
        
        logger.info(f"Processing query: {chat_request.message[:50]}... for session: {chat_request.session_id}")
        result = await medical_chatbot.process_medical_query(
            message=chat_request.message,
            session_id=chat_request.session_id
        )
        
        if not result["success"]:
            logger.error(f"Query processing failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
        
        return {
            "success": True,
            "response": result["response"],
            "urgency_level": result["urgency_level"],
            "emergency": result.get("emergency", False),
            "possible_conditions": result.get("possible_conditions", []),
            "session_id": result["session_id"],
            "timestamp": datetime.now().isoformat(),
            "disclaimer": result.get("disclaimer", "Always consult a healthcare professional."),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", "N/A")
        }
        
    except ValueError as e:
        logger.error(f"Query validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Document upload endpoint"""
    logger.info(f"Upload endpoint hit with session_id: {session_id}, file: {file.filename}")
    try:
        # Validate file
        if not file.filename:
            logger.error("No file provided")
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.allowed_file_types:
            logger.error(f"Unsupported file type: {file_extension}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {settings.allowed_file_types}"
            )
        
        content = await file.read()
        if len(content) > settings.max_upload_size:
            logger.error(f"File too large: {len(content)} bytes")
            raise HTTPException(status_code=400, detail="File too large")
        
        # Sanitize filename to prevent path injection
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        file_path = os.path.join(settings.upload_dir, f"{session_id}_{safe_filename}")
        
        # Ensure upload directory is writable
        try:
            os.makedirs(settings.upload_dir, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"File saved: {file_path}")
        except OSError as e:
            logger.error(f"File write error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # Process document
        result = await medical_chatbot.doc_service.process_document(file_path, file_extension)
        
        if not result["success"]:
            logger.error(f"Document processing failed: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {result['error']}")
        
        logger.info(f"Document processed successfully: {file.filename}")
        return {
            "success": True,
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "file_size": len(content),
            "extracted_text": result["extracted_text"],
            "medical_info": result["medical_info"],
            "analysis": result.get("analysis", {}),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get conversation history for a session"""
    logger.info(f"Session endpoint hit: {session_id}")
    try:
        context = medical_chatbot.context_service.get_context(session_id)
        
        if not context:
            logger.info(f"Session not found: {session_id}")
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
        logger.error(f"Session error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint hit")
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
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/emergency-check")
async def emergency_check(symptoms: str):
    """Quick emergency symptom check"""
    logger.info(f"Emergency check endpoint hit with symptoms: {symptoms[:50]}...")
    try:
        is_emergency = is_emergency_symptom(symptoms)
        
        return {
            "symptoms": symptoms,
            "is_emergency": is_emergency,
            "message": settings.emergency_message if is_emergency else "No immediate emergency detected",
            "urgency_level": "emergency" if is_emergency else "normal"
        }
        
    except Exception as e:
        logger.error(f"Emergency check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Emergency check failed: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request, exc):
    logger.error(f"404 error: {str(exc)}")
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "Check available endpoints at /"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"500 error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Please try again later"}
    )

if __name__ == "__main__":
    logger.info("🚀 Starting Clinical Diagnostics Chatbot API...")
    logger.info(f"🌐 API will be available at: http://localhost:{settings.backend_port}")
    logger.info(f"📚 API documentation at: http://localhost:{settings.backend_port}/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=settings.debug,
        log_level="info"
    )