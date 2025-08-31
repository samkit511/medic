"""
Pytest configuration and shared fixtures for Medical Chatbot tests.

This file contains:
- Test database setup and teardown
- Mock data fixtures  
- API client fixtures
- Authentication fixtures
- HIPAA compliance test utilities
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, Any
from fastapi.testclient import TestClient
from httpx import AsyncClient
import sqlite3

# Import your application modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from main import app
from database import Database
from auth import AuthService, SessionManager
from config import settings


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "security: Security and HIPAA tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def test_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
        tmp_path = tmp_file.name
    yield tmp_path
    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def test_database(test_db_path):
    """Create a test database instance with encryption."""
    db = Database(db_path=test_db_path, encryption_key=None)
    yield db
    db.close()


@pytest.fixture
def test_user_data():
    """Sample user data for testing."""
    return {
        "user_id": "test_user_123",
        "email": "test@medichatbot.com",
        "name": "Test User",
        "picture": "https://example.com/avatar.jpg",
        "email_verified": True
    }


# =============================================================================
# AUTH FIXTURES
# =============================================================================

@pytest.fixture
def auth_service():
    """Create an AuthService instance for testing."""
    return AuthService()


@pytest.fixture
def session_manager():
    """Create a SessionManager instance for testing."""
    return SessionManager()


@pytest.fixture
def valid_jwt_token(auth_service, test_user_data):
    """Create a valid JWT token for testing."""
    return auth_service.create_access_token(data=test_user_data)


@pytest.fixture
def expired_jwt_token(auth_service, test_user_data):
    """Create an expired JWT token for testing."""
    from datetime import timedelta
    return auth_service.create_access_token(
        data=test_user_data,
        expires_delta=timedelta(seconds=-1)  # Already expired
    )


# =============================================================================
# API CLIENT FIXTURES
# =============================================================================

@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def authenticated_client(test_client, valid_jwt_token):
    """Create an authenticated test client."""
    test_client.headers.update({"Authorization": f"Bearer {valid_jwt_token}"})
    return test_client


# =============================================================================
# MOCK DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_medical_query():
    """Sample medical query for testing."""
    return {
        "message": "I have a headache and feel dizzy. What could be wrong?",
        "session_id": "test_session_123",
        "analysis_type": "symptom_check"
    }


@pytest.fixture
def sample_emergency_query():
    """Sample emergency medical query for testing."""
    return {
        "message": "I'm having severe chest pain and difficulty breathing",
        "session_id": "emergency_test_456",
        "analysis_type": "symptom_check"
    }


@pytest.fixture
def sample_chat_history():
    """Sample chat history for testing."""
    return [
        {
            "user_message": "I have a fever",
            "ai_response": "I understand you have a fever. Can you tell me your temperature?",
            "timestamp": "2024-01-01T10:00:00Z",
            "urgency_level": "moderate"
        },
        {
            "user_message": "It's 101.5°F",
            "ai_response": "A fever of 101.5°F is concerning. Are you experiencing any other symptoms?",
            "timestamp": "2024-01-01T10:02:00Z", 
            "urgency_level": "high"
        }
    ]


@pytest.fixture
def sample_upload_file():
    """Create a sample file for upload testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write("Sample medical report content for testing.")
        tmp_file.flush()
        yield tmp_file.name
    # Cleanup
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


# =============================================================================
# MEDICAL DATA FIXTURES
# =============================================================================

@pytest.fixture
def clinical_questions_data():
    """Sample clinical questionnaire data."""
    return {
        "name": "John Doe",
        "age": "35",
        "gender": "Male", 
        "symptom_description": "Persistent headache for 3 days",
        "onset": "3 days ago",
        "duration": "Continuous for 3 days",
        "severity": "7/10",
        "location": "Front of head",
        "associated": "Sensitivity to light",
        "medications": "None",
        "medical_history": "No significant history",
        "allergies": "None known"
    }


@pytest.fixture
def mock_ai_response():
    """Mock AI response for testing."""
    return {
        "success": True,
        "response": "Based on your symptoms, this could be a tension headache or migraine.",
        "urgency_level": "moderate",
        "emergency": False,
        "possible_conditions": ["Tension Headache", "Migraine"],
        "confidence": 0.75,
        "disclaimer": "This analysis is for informational purposes only.",
        "explanation": "Headache with light sensitivity suggests migraine."
    }


# =============================================================================
# HIPAA AND SECURITY FIXTURES
# =============================================================================

@pytest.fixture
def phi_test_data():
    """Protected Health Information test data (de-identified)."""
    return {
        "patient_id": "TEST_001",
        "medical_record_number": "MR12345",
        "date_of_birth": "1990-01-01",
        "ssn": "XXX-XX-1234",  # Masked for testing
        "diagnosis": "Test diagnosis for unit testing",
        "treatment": "Test treatment plan"
    }


@pytest.fixture
def security_headers():
    """Security headers for HIPAA compliance testing."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY", 
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    }


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory for testing."""
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    test_env_vars = {
        "GROQ_API_KEY": "test_groq_key_12345",
        "HUGGINGFACE_API_TOKEN": "test_hf_token_67890",
        "JWT_SECRET_KEY": "test_jwt_secret_key_for_testing_only",
        "DATABASE_ENCRYPTION_KEY": "test_db_encryption_key"
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_env_vars


# =============================================================================
# PERFORMANCE TESTING FIXTURES  
# =============================================================================

@pytest.fixture
def performance_test_data():
    """Generate test data for performance testing."""
    return {
        "large_text": "This is a long medical text. " * 1000,  # Simulate large input
        "multiple_queries": [f"Test query {i}" for i in range(100)],
        "concurrent_users": 50
    }


# =============================================================================
# CLEANUP AND TEARDOWN
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    # Cleanup logic can be added here if needed
    pass


def pytest_sessionfinish(session, exitstatus):
    """Cleanup after all tests complete."""
    # Clean up any global test artifacts
    pass
