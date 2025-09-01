import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

load_dotenv()  # Load .env file

class Settings(BaseSettings):
    """Application configuration"""
    
    # Application Info
    app_name: str = "Clinical Diagnostics Chatbot"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # API Configuration
    backend_port: int = 8000
    backend_https_port: int = 8443
    frontend_port: int = 8501
    frontend_https_port: int = 3443
    
    # SSL Configuration
    ssl_cert_file: str = "certs/localhost.crt"
    ssl_key_file: str = "certs/localhost.key"
    ssl_enabled: bool = True
    
    # AI API Keys
    huggingface_api_token: str = os.getenv("huggingface_api_token", "")
    groq_api_key: str = os.getenv("groq_api_key", "")
    
    # JWT Authentication - HIPAA Compliant
    jwt_secret_key: str = os.getenv("jwt_secret_key", "fallback-secret-key-for-development-only-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30  # HIPAA recommended session timeout
    jwt_refresh_token_expire_days: int = 7     # Reduced for security
    
    # Additional settings from .env file
    database_encryption_key: str = os.getenv("database_encryption_key", "")
    ssl_cert_path: str = "ssl/server.crt"
    ssl_key_path: str = "ssl/server.key"
    log_level: str = "INFO"
    audit_log_enabled: bool = True
    require_mfa: bool = False
    
    # Database settings
    database_url: str = "sqlite:///data/medical_chatbot.db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    # Backup settings
    backup_enabled: bool = True
    backup_schedule: str = "daily"
    backup_retention_days: int = 30
    backup_location: str = "backups/"
    
    # Monitoring settings
    monitoring_enabled: bool = True
    monitoring_check_interval: int = 30
    alert_cooldown_minutes: int = 60
    
    # MFA settings
    mfa_issuer_name: str = "Medical_Chatbot_HIPAA"
    mfa_backup_codes_count: int = 10
    
    # SMTP settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    
    # HIPAA Security Settings
    phi_encryption_enabled: bool = True
    audit_logging_enabled: bool = True
    session_timeout_minutes: int = 30
    max_failed_login_attempts: int = 3
    password_min_length: int = 12
    require_mfa: bool = False  # Set to True for production
    
    # Google OAuth
    google_client_id: str = os.getenv("google_client_id", "")
    
    # CORS Configuration
    allowed_origins: List[str] = [
        "http://localhost:8501", "http://localhost:3000", "http://localhost:5173",
        "https://localhost:3443", "https://localhost:3000", "https://localhost:5173"
    ]
    
    # Medical Configuration
    max_upload_size: int = 10485760  # 10MB
    allowed_file_types: List[str] = ["pdf", "txt", "jpg", "png", "docx"]
    emergency_keywords: List[str] = [
        "chest pain", "difficulty breathing", "severe bleeding",
        "unconscious", "stroke", "heart attack", "poisoning"
    ]
    
    # AI Models - Updated to use accessible HuggingFace models
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_medical_model: str = "microsoft/DialoGPT-medium"  # Accessible conversational model for medical queries
    hf_text_generation_model: str = "gpt2"  # Reliable text generation model
    hf_biobert_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"  # BioBERT for medical NLI
    groq_model: str = "llama-3.1-8b-instant"  # Updated model after deprecation of llama3-8b-8192
    
    # Database Configuration
    database_path: str = "data/chatbot.db"
    database_encryption_enabled: bool = True
    database_key_file: str = "data/db_key.key"
    database_backup_dir: str = "data/backups"
    
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
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="allow"  # Allow extra fields from .env
    )

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