#!/usr/bin/env python3
"""
HTTPS Server Startup Script for Medical Chatbot
Starts the FastAPI backend with SSL/HTTPS support
"""

import uvicorn
import os
import sys
from pathlib import Path

def check_ssl_certificates():
    """Check if SSL certificates exist"""
    ssl_dir = Path(__file__).parent / "ssl"
    cert_file = ssl_dir / "server.crt"
    key_file = ssl_dir / "server.key"
    
    if not ssl_dir.exists():
        print("❌ SSL directory not found!")
        print("Run 'python generate_ssl.py' first to create SSL certificates.")
        return False
    
    if not cert_file.exists() or not key_file.exists():
        print("❌ SSL certificates not found!")
        print("Run 'python generate_ssl.py' to create SSL certificates.")
        return False
    
    print("✅ SSL certificates found")
    return True

def start_https_server():
    """Start the FastAPI server with HTTPS"""
    
    if not check_ssl_certificates():
        sys.exit(1)
    
    ssl_dir = Path(__file__).parent / "ssl"
    cert_file = str(ssl_dir / "server.crt")
    key_file = str(ssl_dir / "server.key")
    
    print("🚀 Starting Medical Chatbot HTTPS Server...")
    print("📍 Server will be available at: https://localhost:8000")
    print("📋 API documentation: https://localhost:8000/docs")
    print("🔒 SSL/TLS encryption: ENABLED")
    print("\n⚠️  Note: You may see a security warning in your browser")
    print("   Click 'Advanced' -> 'Proceed to localhost (unsafe)' to continue")
    print("\n🛑 Press Ctrl+C to stop the server\n")
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting HTTPS server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_https_server()
