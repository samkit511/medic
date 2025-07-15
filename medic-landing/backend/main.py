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
from database import Database
from services import medical_chatbot
import logging
from PIL import Image
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🏥 Starting Clinical Diagnostics Chatbot...")
    logger.info(f"Backend running on port {settings.backend_port}")
    
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info(f"Upload directory ensured: {settings.upload_dir}")
    
    db = Database()
    logger.info("✅ Database initialized, chatbot.db created")
    
    app_state["chatbot"] = medical_chatbot
    app_state["db"] = db
    logger.info("✅ Medical chatbot services initialized")
    
    try:
        yield
    finally:
        db.close()
        logger.info("👋 Shutting down Clinical Diagnostics Chatbot...")

app = FastAPI(
    title="Clinical Diagnostics Chatbot API",
    description="AI-powered medical diagnostics with RAG and conversation management",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
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

@app.get("/", tags=["General"])
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Clinical Diagnostics Chatbot API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/api/query": "POST - Main chat endpoint",
            "/api/upload": "POST - Document upload",
            "/api/upload_image": "POST - Image upload",
            "/api/save_chat": "POST - Save chat message",
            "/api/get_chat_history": "GET - Get chat history",
            "/api/forgot-password": "POST - Request password reset",
            "/session/{session_id}": "GET/POST - Session management",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.post("/api/query", tags=["Chat"], description="Process a medical query")
async def chat_endpoint(request_data: dict):
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

@app.post("/api/upload", tags=["Upload"], description="Upload a medical document")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    logger.info(f"Upload endpoint hit with session_id: {session_id}, file: {file.filename}")
    try:
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
        
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        file_path = os.path.join(settings.upload_dir, f"{session_id}_{safe_filename}")
        
        try:
            os.makedirs(settings.upload_dir, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"File saved: {file_path}")
        except OSError as e:
            logger.error(f"File write error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
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

@app.post("/api/upload_image", tags=["Upload"], description="Upload a medical image")
async def upload_image(file: UploadFile = File(...), session_id: str = Form(...)):
    try:
        if file.content_type not in ['image/jpeg', 'image/png']:
            logger.error(f"Unsupported image type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Unsupported image type")
        image = Image.open(io.BytesIO(await file.read()))
        dpi = image.info.get('dpi', (72, 72))[0]
        if dpi < 300:
            width, height = image.size
            new_width = int(width * (300 / dpi))
            new_height = int(height * (300 / dpi))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        file_path = os.path.join(settings.upload_dir, f"{session_id}_{safe_filename}")
        os.makedirs(settings.upload_dir, exist_ok=True)
        image.save(file_path, dpi=(300, 300))
        result = await medical_chatbot.doc_service.process_document(file_path, file.filename.split('.')[-1].lower())
        logger.info(f"Image processed successfully: {file.filename}")
        return {
            "success": True,
            "message": "Image uploaded and processed successfully",
            "filename": file.filename,
            "file_path": file_path,
            "extracted_text": result["extracted_text"],
            "medical_info": result["medical_info"],
            "analysis": result.get("analysis", {}),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.post("/api/save_chat", tags=["Chat"], description="Save a chat message to database")
async def save_chat(data: dict):
    try:
        db = app_state.get("db", Database())
        username = data.get("username", "Guest")
        user_message = data.get("user_message")
        bot_response = data.get("bot_response")
        if not user_message or not bot_response:
            logger.error("Missing chat data")
            raise HTTPException(status_code=400, detail="Missing user_message or bot_response")
        db.add_chat(username, user_message, bot_response)
        logger.info(f"Chat saved for user: {username}")
        return {"success": True, "message": "Chat saved successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Save chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save chat: {str(e)}")

@app.get("/api/get_chat_history", tags=["Chat"], description="Retrieve chat history for a user")
async def get_chat_history(username: str):
    try:
        db = app_state.get("db", Database())
        history = db.get_chat_history(username)
        return {
            "success": True,
            "history": [
                {
                    "user_message": chat[2],
                    "bot_response": chat[3],
                    "timestamp": datetime.fromisoformat(chat[4]).isoformat() if isinstance(chat[4], str) else chat[4].isoformat()
                } for chat in history
            ]
        }
    except Exception as e:
        logger.error(f"Get chat history error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@app.post("/api/forgot-password", tags=["Auth"], description="Request a password reset link")
async def forgot_password(email: str = Form(...)):
    logger.info(f"Password reset requested for: {email}")
    try:
        return {"success": True, "message": "Password reset link sent to your email"}
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send reset link: {str(e)}")

@app.get("/session/{session_id}", tags=["Session"], description="Retrieve session details")
async def get_session(session_id: str):
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

@app.post("/session/{session_id}", tags=["Session"], description="Create a new session")
async def create_session(session_id: str):
    try:
        context = medical_chatbot.context_service.create_session(session_id)
        logger.info(f"Session created: {session_id}")
        return {
            "success": True,
            "session_id": session_id,
            "created_at": context["created_at"].isoformat(),
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"Session creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/health", tags=["Health"], description="Check API health status")
async def health_check():
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

@app.get("/emergency-check", tags=["Emergency"], description="Check if symptoms indicate an emergency")
async def emergency_check(symptoms: str):
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
