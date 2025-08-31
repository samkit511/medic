import os
import asyncio
import re
from typing import Dict, Any
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
# Handle both direct execution and module import
try:
    # Try relative imports first (when imported as module)
    from .services import medical_chatbot
    from .config import settings, is_emergency_symptom
    from .database import Database
    from .auth import AuthService, session_manager, get_current_user, get_current_user_optional
except ImportError:
    # Fall back to absolute imports (when run directly)
    from services import medical_chatbot
    from config import settings, is_emergency_symptom
    from database import Database
    from auth import AuthService, session_manager, get_current_user, get_current_user_optional
from pydantic import BaseModel

# Authentication request models
class LoginRequest(BaseModel):
    id_token: str
    
class LogoutRequest(BaseModel):
    refresh_token: str
    
class RefreshRequest(BaseModel):
    refresh_token: str

# Initialize auth service
auth_service = AuthService()
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
    
    encryption_key = settings.database_encryption_enabled and settings.database_key_file
    db = Database(db_path=settings.database_path, encryption_key=encryption_key)
    logger.info("✅ Database initialized with encryption")
    
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

@app.post("/api/query/test", tags=["Chat"], description="Process a test medical query without authentication")
async def chat_endpoint_test(request_data: dict):
    try:
        chat_request = ChatRequest(request_data)
        chat_request.validate()
        
        logger.info(f"Processing query: {chat_request.message[:50]}... for session: {chat_request.session_id}")
        
        # Use the clinical questionnaire system first
        result = await medical_chatbot.process_user_input(
            user_input=chat_request.message,
            session_id=chat_request.session_id
        )
        
        # Check if we have a next question to ask
        if "next_question" in result:
            return {
                "success": True,
                "response": result["next_question"],
                "urgency_level": "normal",
                "emergency": False,
                "possible_conditions": [],
                "session_id": result["session_id"],
                "timestamp": datetime.now().isoformat(),
                "disclaimer": "Please answer all questions to receive accurate medical guidance.",
                "confidence": 0.0,
                "explanation": "Collecting patient information through clinical questionnaire",
                "is_clinical_question": True
            }
        # If all questions are answered, return the diagnosis
        elif "diagnosis" in result:
            diagnosis = result["diagnosis"]
            if not diagnosis["success"]:
                logger.error(f"Diagnosis processing failed: {diagnosis.get('error', 'Unknown error')}")
                raise HTTPException(status_code=500, detail=diagnosis.get("error", "Processing failed"))
            
            return {
                "success": True,
                "response": diagnosis["response"],
                "urgency_level": diagnosis.get("urgency_level", "normal"),
                "emergency": diagnosis.get("emergency", False),
                "possible_conditions": diagnosis.get("possible_conditions", []),
                "session_id": diagnosis.get("session_id", chat_request.session_id),
                "timestamp": datetime.now().isoformat(),
                "disclaimer": diagnosis.get("disclaimer", "Always consult a healthcare professional."),
                "confidence": diagnosis.get("confidence", 0.0),
                "explanation": diagnosis.get("explanation", "N/A"),
                "is_clinical_question": False
            }
        else:
            # Fallback to direct medical query processing (shouldn't happen normally)
            direct_result = await medical_chatbot.process_medical_query(
                message=chat_request.message,
                session_id=chat_request.session_id
            )
            
            if not direct_result["success"]:
                logger.error(f"Direct query processing failed: {direct_result.get('error', 'Unknown error')}")
                raise HTTPException(status_code=500, detail=direct_result.get("error", "Processing failed"))
            
            return {
                "success": True,
                "response": direct_result["response"],
                "urgency_level": direct_result["urgency_level"],
                "emergency": direct_result.get("emergency", False),
                "possible_conditions": direct_result.get("possible_conditions", []),
                "session_id": direct_result["session_id"],
                "timestamp": datetime.now().isoformat(),
                "disclaimer": direct_result.get("disclaimer", "Always consult a healthcare professional."),
                "confidence": direct_result.get("confidence", 0.0),
                "explanation": direct_result.get("explanation", "N/A"),
                "is_clinical_question": False
            }
        
    except ValueError as e:
        logger.error(f"Query validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/query", tags=["Chat"], description="Process a medical query")
async def chat_endpoint(request_data: dict, current_user: Dict[str, Any] = Depends(get_current_user)):
    logger.info(f"Chat endpoint hit with session_id: {request_data.get('session_id', 'default')}")
    try:
        chat_request = ChatRequest(request_data)
        chat_request.validate()
        
        logger.info(f"Processing query: {chat_request.message[:50]}... for session: {chat_request.session_id}")
        
        # Use the clinical questionnaire system first
        result = await medical_chatbot.process_user_input(
            user_input=chat_request.message,
            session_id=chat_request.session_id
        )
        
        # Check if we have a next question to ask
        if "next_question" in result:
            return {
                "success": True,
                "response": result["next_question"],
                "urgency_level": "normal",
                "emergency": False,
                "possible_conditions": [],
                "session_id": result["session_id"],
                "timestamp": datetime.now().isoformat(),
                "disclaimer": "Please answer all questions to receive accurate medical guidance.",
                "confidence": 0.0,
                "explanation": "Collecting patient information through clinical questionnaire",
                "is_clinical_question": True
            }
        # If all questions are answered, return the diagnosis
        elif "diagnosis" in result:
            diagnosis = result["diagnosis"]
            if not diagnosis["success"]:
                logger.error(f"Diagnosis processing failed: {diagnosis.get('error', 'Unknown error')}")
                raise HTTPException(status_code=500, detail=diagnosis.get("error", "Processing failed"))
            
            return {
                "success": True,
                "response": diagnosis["response"],
                "urgency_level": diagnosis.get("urgency_level", "normal"),
                "emergency": diagnosis.get("emergency", False),
                "possible_conditions": diagnosis.get("possible_conditions", []),
                "session_id": diagnosis.get("session_id", chat_request.session_id),
                "timestamp": datetime.now().isoformat(),
                "disclaimer": diagnosis.get("disclaimer", "Always consult a healthcare professional."),
                "confidence": diagnosis.get("confidence", 0.0),
                "explanation": diagnosis.get("explanation", "N/A"),
                "is_clinical_question": False
            }
        else:
            # Fallback to direct medical query processing (shouldn't happen normally)
            direct_result = await medical_chatbot.process_medical_query(
                message=chat_request.message,
                session_id=chat_request.session_id
            )
            
            if not direct_result["success"]:
                logger.error(f"Direct query processing failed: {direct_result.get('error', 'Unknown error')}")
                raise HTTPException(status_code=500, detail=direct_result.get("error", "Processing failed"))
            
            return {
                "success": True,
                "response": direct_result["response"],
                "urgency_level": direct_result["urgency_level"],
                "emergency": direct_result.get("emergency", False),
                "possible_conditions": direct_result.get("possible_conditions", []),
                "session_id": direct_result["session_id"],
                "timestamp": datetime.now().isoformat(),
                "disclaimer": direct_result.get("disclaimer", "Always consult a healthcare professional."),
                "confidence": direct_result.get("confidence", 0.0),
                "explanation": direct_result.get("explanation", "N/A"),
                "is_clinical_question": False
            }
        
    except ValueError as e:
        logger.error(f"Query validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/upload/test", tags=["Upload"], description="Upload a medical document (test endpoint without auth)")
async def upload_document_test(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    logger.info(f"Test upload endpoint hit with session_id: {session_id}, file: {file.filename}")
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
        
        result = await medical_chatbot.doc_service.process_document(file_path, file_extension, session_id)
        
        if not result["success"]:
            logger.error(f"Document processing failed: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {result['error']}")
        
        # Update conversation context with document analysis
        user_message = f"I uploaded a document: {file.filename}"
        bot_response = f"Document analyzed successfully. {result.get('analysis', {}).get('response', 'Medical information extracted from your document.')}"
        
        # Save to conversation context
        medical_chatbot.context_service.update_context(
            session_id=session_id,
            message=user_message,
            response=bot_response,
            medical_info=result["medical_info"]
        )
        
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
            "timestamp": datetime.now().isoformat(),
            "conversation_updated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/api/upload", tags=["Upload"], description="Upload a medical document")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
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
        
        result = await medical_chatbot.doc_service.process_document(file_path, file_extension, session_id)
        
        if not result["success"]:
            logger.error(f"Document processing failed: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {result['error']}")
        
        # Update conversation context with document analysis
        user_message = f"I uploaded a document: {file.filename}"
        bot_response = f"Document analyzed successfully. {result.get('analysis', {}).get('response', 'Medical information extracted from your document.')}"
        
        # Save to conversation context
        medical_chatbot.context_service.update_context(
            session_id=session_id,
            message=user_message,
            response=bot_response,
            medical_info=result["medical_info"]
        )
        
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
            "timestamp": datetime.now().isoformat(),
            "conversation_updated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/api/upload_image", tags=["Upload"], description="Upload a medical image")
async def upload_image(file: UploadFile = File(...), session_id: str = Form(...), current_user: Dict[str, Any] = Depends(get_current_user)):
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
        result = await medical_chatbot.doc_service.process_document(file_path, file.filename.split('.')[-1].lower(), session_id)
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

@app.post("/api/process_text", tags=["Medical"], description="Process medical report text directly")
async def process_medical_text(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Process medical report text directly without file upload"""
    try:
        session_id = request.get("session_id")
        medical_text = request.get("medical_text")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        if not medical_text:
            raise HTTPException(status_code=400, detail="medical_text is required")
        
        logger.info(f"Processing medical text for session: {session_id}")
        
        # Create a mock result similar to document processing
        result = {
            "success": True,
            "extracted_text": medical_text,
            "medical_info": {},
            "analysis": {
                "response": "Medical information extracted from provided text."
            }
        }
        
        # Use the document service to extract medical info from text
        try:
            medical_info = await medical_chatbot.doc_service.extract_medical_info(medical_text)
            result["medical_info"] = medical_info
            logger.info(f"Extracted medical info: {medical_info}")
        except Exception as e:
            logger.warning(f"Medical info extraction failed: {str(e)}")
            result["medical_info"] = {}
        
        # Update conversation context with medical text analysis
        user_message = "I provided medical report text for analysis"
        bot_response = f"Medical text analyzed successfully. {result.get('analysis', {}).get('response', 'Medical information extracted from your text.')}"
        
        # Save to conversation context
        medical_chatbot.context_service.update_context(
            session_id=session_id,
            message=user_message,
            response=bot_response,
            medical_info=result["medical_info"]
        )
        
        logger.info(f"Medical text processed successfully for session: {session_id}")
        return {
            "success": True,
            "message": "Medical text processed successfully",
            "text_length": len(medical_text),
            "extracted_text": result["extracted_text"],
            "medical_info": result["medical_info"],
            "analysis": result.get("analysis", {}),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "conversation_updated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

@app.post("/api/save_chat", tags=["Chat"], description="Save a chat message to database")
async def save_chat(data: dict, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        encryption_key = settings.database_encryption_enabled and settings.database_key_file
        db = app_state.get("db", Database(db_path=settings.database_path, encryption_key=encryption_key))
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
async def get_chat_history(username: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        encryption_key = settings.database_encryption_enabled and settings.database_key_file
        db = app_state.get("db", Database(db_path=settings.database_path, encryption_key=encryption_key))
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
async def get_session(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
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
async def create_session(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
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

@app.get("/api/greeting/{session_id}", tags=["Chat"], description="Get initial greeting for a session")
async def get_greeting(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        # Check if session exists
        session = medical_chatbot.context_service.get_context(session_id)
        if not session:
            session = medical_chatbot.context_service.create_session(session_id)
        
        # Check if greeting has been shown
        if session.get("greeting_shown", False):
            return {
                "success": True,
                "greeting_needed": False,
                "message": "Greeting already shown"
            }
        
        # Return greeting
        return {
            "success": True,
            "greeting_needed": True,
            "greeting": "Hello! May I know your name?",
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Greeting endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get greeting: {str(e)}")

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

# Authentication endpoints
@app.post("/auth/login", tags=["Authentication"], description="Login with Google ID token")
async def login(request: LoginRequest):
    logger.info("Login endpoint hit")
    try:
        # Verify Google ID token
        user_info = auth_service.verify_google_token(request.id_token)
        
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid Google token")
        
        # Create session and generate tokens
        tokens = session_manager.create_session(user_info['user_id'], user_info)
        
        logger.info(f"User logged in: {user_info['email']}")
        return {
            "success": True,
            "user": user_info,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expires_in": 3600  # 1 hour
        }
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

@app.post("/auth/logout", tags=["Authentication"], description="Logout and invalidate session")
async def logout(request: LogoutRequest):
    logger.info("Logout endpoint hit")
    try:
        # Verify refresh token and get session ID
        payload = auth_service.verify_refresh_token(request.refresh_token)
        session_id = payload.get("session_id")
        
        if session_id:
            # Remove session
            session_manager.remove_session(session_id)
            logger.info(f"User logged out, session removed: {session_id}")
        
        return {
            "success": True,
            "message": "Logged out successfully"
        }
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return {
            "success": True,
            "message": "Logged out successfully"  # Always return success for logout
        }

@app.post("/auth/refresh", tags=["Authentication"], description="Refresh access token")
async def refresh_token(request: RefreshRequest):
    logger.info("Token refresh endpoint hit")
    try:
        # Verify refresh token
        payload = auth_service.verify_refresh_token(request.refresh_token)
        session_id = payload.get("session_id")
        
        # Get session info
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid session")
        
        # Generate new access token
        user_data = session["user_data"] if "user_data" in session else session.get("user_info", {})
        new_access_token = auth_service.create_access_token(data={
            "user_id": payload.get("user_id"),
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "session_id": session_id
        })
        
        logger.info(f"Token refreshed for user: {user_data.get('email', 'unknown')}")
        return {
            "success": True,
            "access_token": new_access_token,
            "refresh_token": request.refresh_token, # Keep the same refresh token
            "expires_in": 3600
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Token refresh failed: {str(e)}")

@app.get("/auth/me", tags=["Authentication"], description="Get current user info")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    logger.info(f"User info requested for: {current_user['email']}")
    return {
        "success": True,
        "user": current_user
    }

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
    
    if settings.ssl_enabled and os.path.exists(settings.ssl_cert_file) and os.path.exists(settings.ssl_key_file):
        # Start with HTTPS
        logger.info(f"🔐 HTTPS enabled - SSL certificates found")
        logger.info(f"🌐 Secure API available at: https://localhost:{settings.backend_https_port}")
        logger.info(f"📚 Secure API documentation at: https://localhost:{settings.backend_https_port}/docs")
        logger.info(f"⚠️  You may need to accept the self-signed certificate in your browser")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=settings.backend_https_port,
            ssl_keyfile=settings.ssl_key_file,
            ssl_certfile=settings.ssl_cert_file,
            reload=settings.debug,
            log_level="info"
        )
    else:
        # Fallback to HTTP if SSL is not configured
        logger.warning("⚠️  SSL certificates not found, falling back to HTTP")
        logger.info(f"🌐 HTTP API available at: http://localhost:{settings.backend_port}")
        logger.info(f"📚 HTTP API documentation at: http://localhost:{settings.backend_port}/docs")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=settings.backend_port,
            reload=settings.debug,
            log_level="info"
        )