import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application configuration"""
    
    # Application Info
    app_name: str = "Clinical Diagnostics Chatbot"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # API Configuration
    backend_port: int = 8000
    frontend_port: int = 8501
    
    # AI API Keys
    huggingface_api_token: str = ""
    groq_api_key: str = ""
    
    # CORS Configuration
    allowed_origins: List[str] = ["http://localhost:8501", "http://localhost:3000", "http://localhost:5173"]
    
    # Medical Configuration
    max_upload_size: int = 10485760  # 10MB
    allowed_file_types: List[str] = ["pdf", "txt", "jpg", "png"]
    emergency_keywords: List[str] = [
        "chest pain", "difficulty breathing", "severe bleeding",
        "unconscious", "stroke", "heart attack", "poisoning"
    ]
    
    # AI Models
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_medical_model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    groq_model: str = "llama3-8b-8192"
    
    # File Storage
    upload_dir: str = "data/uploads"
    documents_dir: str = "data/documents"
    
    # Medical Response Settings
    max_response_tokens: int = 500
    medical_temperature: float = 0.3
    
    # Medical Disclaimers
    medical_disclaimer: str = (
        "WARNING: This analysis is for informational purposes only. "
        "Always consult qualified healthcare providers for medical advice."
    )
    
    emergency_message: str = (
        "Medical Advice: Seek immediate medical attention! "
        "Call emergency services or go to the nearest emergency department."
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = "utf-8"

settings = Settings()

def get_api_keys():
    return {
        "groq": settings.groq_api_key,
        "huggingface": settings.huggingface_api_token
    }

def is_emergency_symptom(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword.strip().lower() in text_lower for keyword in settings.emergency_keywords)

def validate_environment():
    api_keys = get_api_keys()
    has_groq = bool(api_keys["groq"].strip())
    has_hf = bool(api_keys["huggingface"].strip())
    
    if not (has_groq or has_hf):
        raise ValueError("Need either GROQ_API_KEY or HUGGINGFACE_API_TOKEN")
    
    return {
        "groq_available": has_groq,
        "huggingface_available": has_hf
    }

if __name__ == "__main__":
    print("Clinical Diagnostics Chatbot Configuration")
    print(f"App: {settings.app_name} v{settings.app_version}")
    
    try:
        env_status = validate_environment()
        print("API Status:")
        for service, available in env_status.items():
            status = "✅ Available" if available else "❌ Missing"
            print(f"  {service}: {status}")
    except ValueError as e:
        print(f"❌ Error: {e}")