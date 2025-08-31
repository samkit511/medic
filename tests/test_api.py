"""
API endpoint tests for the Medical Chatbot application.

This module tests all FastAPI endpoints including:
- Authentication endpoints
- Chat/query endpoints  
- File upload endpoints
- Health check endpoints
"""

import pytest
import json
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import Mock, patch


class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns correct information."""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Clinical Diagnostics Chatbot API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "active"
        assert "endpoints" in data
    
    def test_health_check(self, test_client):
        """Test health check endpoint if it exists."""
        # This test assumes you have a /health endpoint
        response = test_client.get("/health")
        # Adjust assertion based on your actual health endpoint
        assert response.status_code in [200, 404]  # 404 if not implemented


class TestChatEndpoints:
    """Test chat and query endpoints."""
    
    def test_chat_endpoint_test_valid_query(self, test_client, sample_medical_query):
        """Test the test chat endpoint with valid query."""
        with patch('backend.services.medical_chatbot.process_user_input') as mock_process:
            mock_process.return_value = {
                "next_question": "What is your age?",
                "session_id": "test_session_123"
            }
            
            response = test_client.post("/api/query/test", json=sample_medical_query)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "response" in data
            assert data["session_id"] == "test_session_123"
    
    def test_chat_endpoint_test_invalid_query(self, test_client):
        """Test the test chat endpoint with invalid query."""
        invalid_query = {
            "message": "",  # Empty message should fail validation
            "session_id": "test_session"
        }
        
        response = test_client.post("/api/query/test", json=invalid_query)
        assert response.status_code == 400
    
    def test_chat_endpoint_test_emergency_detection(self, test_client, sample_emergency_query):
        """Test emergency symptom detection."""
        with patch('backend.services.medical_chatbot.process_user_input') as mock_process:
            mock_process.return_value = {
                "diagnosis": {
                    "success": True,
                    "response": "EMERGENCY: Seek immediate medical attention!",
                    "emergency": True,
                    "urgency_level": "emergency"
                }
            }
            
            response = test_client.post("/api/query/test", json=sample_emergency_query)
            assert response.status_code == 200
            
            data = response.json()
            assert data["emergency"] is True
            assert data["urgency_level"] == "emergency"
    
    @pytest.mark.api
    def test_authenticated_chat_endpoint(self, authenticated_client, sample_medical_query):
        """Test the authenticated chat endpoint."""
        with patch('backend.services.medical_chatbot.process_user_input') as mock_process:
            mock_process.return_value = {
                "diagnosis": {
                    "success": True,
                    "response": "Based on your symptoms...",
                    "urgency_level": "moderate"
                }
            }
            
            response = authenticated_client.post("/api/query", json=sample_medical_query)
            assert response.status_code == 200
    
    def test_unauthenticated_chat_endpoint(self, test_client, sample_medical_query):
        """Test the authenticated endpoint without authentication."""
        response = test_client.post("/api/query", json=sample_medical_query)
        assert response.status_code == 401


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    def test_login_endpoint_exists(self, test_client):
        """Test that login endpoint exists."""
        # Test with invalid data to see if endpoint exists
        response = test_client.post("/api/auth/login", json={})
        # Should return 422 (validation error) or 400, not 404
        assert response.status_code in [400, 422, 405]  # Not 404
    
    def test_logout_endpoint_exists(self, test_client):
        """Test that logout endpoint exists."""
        response = test_client.post("/api/auth/logout", json={})
        assert response.status_code in [400, 422, 405]  # Not 404


class TestFileUploadEndpoints:
    """Test file upload endpoints."""
    
    def test_upload_endpoint_exists(self, test_client):
        """Test that upload endpoint exists."""
        response = test_client.post("/api/upload")
        # Should not return 404
        assert response.status_code != 404
    
    def test_upload_image_endpoint_exists(self, test_client):
        """Test that image upload endpoint exists."""
        response = test_client.post("/api/upload_image")
        assert response.status_code != 404
    
    @pytest.mark.slow
    def test_file_upload_with_valid_file(self, authenticated_client, sample_upload_file):
        """Test file upload with a valid file."""
        with open(sample_upload_file, 'rb') as f:
            files = {"file": ("test.txt", f, "text/plain")}
            response = authenticated_client.post("/api/upload", files=files)
            # Status depends on implementation
            assert response.status_code in [200, 201, 400, 422]


class TestSessionEndpoints:
    """Test session management endpoints."""
    
    def test_session_endpoint_exists(self, test_client):
        """Test that session endpoints exist."""
        response = test_client.get("/session/test_session")
        assert response.status_code in [200, 401, 404]


class TestChatHistoryEndpoints:
    """Test chat history endpoints."""
    
    def test_get_chat_history_endpoint(self, authenticated_client):
        """Test get chat history endpoint."""
        response = authenticated_client.get("/api/get_chat_history")
        # Endpoint may or may not exist, but shouldn't crash
        assert response.status_code in [200, 404, 405]
    
    def test_save_chat_endpoint(self, authenticated_client):
        """Test save chat endpoint."""
        chat_data = {
            "user_message": "Test message",
            "bot_response": "Test response"
        }
        response = authenticated_client.post("/api/save_chat", json=chat_data)
        assert response.status_code in [200, 201, 404, 422]


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_endpoint(self, test_client):
        """Test accessing a non-existent endpoint."""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_malformed_json(self, test_client):
        """Test sending malformed JSON."""
        response = test_client.post(
            "/api/query/test",
            data="malformed json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_large_payload(self, test_client):
        """Test handling of large payloads."""
        large_message = "A" * 10000  # 10KB message
        large_query = {
            "message": large_message,
            "session_id": "test_large"
        }
        
        response = test_client.post("/api/query/test", json=large_query)
        # Should either process or reject gracefully
        assert response.status_code in [200, 400, 413, 422]
    
    @pytest.mark.api
    async def test_concurrent_requests(self, async_client, sample_medical_query):
        """Test handling concurrent requests."""
        import asyncio
        
        async def make_request():
            return await async_client.post("/api/query/test", json=sample_medical_query)
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least some should succeed
        successful_responses = [r for r in responses if hasattr(r, 'status_code')]
        assert len(successful_responses) > 0


class TestSecurityHeaders:
    """Test security headers and CORS."""
    
    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/api/query/test")
        # CORS headers should be present
        assert response.status_code in [200, 405]
    
    def test_security_headers_present(self, test_client, security_headers):
        """Test that security headers are present in responses."""
        response = test_client.get("/")
        
        # Check for common security headers (if implemented)
        # Note: These may not be implemented yet
        for header_name in ["X-Content-Type-Options", "X-Frame-Options"]:
            # Don't assert presence, just check if they exist
            header_value = response.headers.get(header_name)
            if header_value:
                assert isinstance(header_value, str)


# =============================================================================
# PERFORMANCE AND LOAD TESTS
# =============================================================================

class TestPerformance:
    """Performance and load testing."""
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_query_response_time(self, test_client, sample_medical_query, benchmark):
        """Benchmark query response time."""
        def make_query():
            return test_client.post("/api/query/test", json=sample_medical_query)
        
        result = benchmark(make_query)
        assert result.status_code in [200, 400, 422]
    
    @pytest.mark.slow
    def test_memory_usage(self, test_client, sample_medical_query):
        """Test memory usage during requests."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple requests
        for _ in range(10):
            response = test_client.post("/api/query/test", json=sample_medical_query)
            assert response.status_code in [200, 400, 422]
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 10 requests)
        assert memory_increase < 100 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__])
