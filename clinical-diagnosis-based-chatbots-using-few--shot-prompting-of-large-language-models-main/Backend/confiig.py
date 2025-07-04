
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application configuration - Simplified for Groq + HuggingFace only"""
    
 
    app_name: str = "Clinical Diagnostics Chatbot"
    app_version: str = "1.0.0"
    debug: bool = True
    
    
    backend_port: int = 8000
    frontend_port: int = 8501
    
    huggingface_api_token: str = ""
    groq_api_key: str = ""
    
    max_upload_size: int = 10485760  # 10MB
    allowed_file_types: List[str] = ["pdf", "txt", "docx"]
    emergency_keywords: List[str] = [
        "chest pain", "difficulty breathing", "severe bleeding",
        "unconscious", "stroke", "heart attack", "poisoning"
    ]
    
    
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_medical_model: str = "microsoft/DialoGPT-large"
    groq_model: str = "llama3-8b-8192"
    
    
    upload_dir: str = "data/uploads"
    documents_dir: str = "data/documents"
    
    
    max_response_tokens: int = 500
    medical_temperature: float = 0.3
    
    
    medical_disclaimer: str = (
        "WARNING: This analysis is for informational purposes only. "
        "Always consult qualified healthcare professionals for medical advice."
    )
    
    emergency_message: str = (
        "EMERGENCY: Seek immediate medical attention! "
        "Call emergency services or go to the nearest emergency room."
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

def get_api_keys():
    return {
        "groq": settings.groq_api_key,
        "huggingface": settings.huggingface_api_token
    }

def is_emergency_symptom(text: str) -> bool:
    """Check if text contains emergency keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in settings.emergency_keywords)

def validate_environment():
    """Validate that we have the API keys we need"""
    api_keys = get_api_keys()
    has_groq = bool(api_keys["groq"])
    has_hf = bool(api_keys["huggingface"])
    
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
            status = "✅ Ready" if available else "❌ Missing"
            print(f"  {service}: {status}")
    except ValueError as e:
        print(f"❌ Error: {e}")
