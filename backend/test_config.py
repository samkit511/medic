#!/usr/bin/env python3
"""Test script to validate API key configuration"""

import os
from config import settings, get_api_keys, validate_environment

def test_config():
    print("=" * 50)
    print("API Configuration Test")
    print("=" * 50)
    
    # Test environment variables directly
    print("\nDirect Environment Variables:")
    print(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY', 'NOT SET')[:20]}...")
    print(f"HUGGINGFACE_API_TOKEN: {os.getenv('HUGGINGFACE_API_TOKEN', 'NOT SET')[:20]}...")
    
    # Test config loading
    print("\nConfig Settings:")
    print(f"groq_api_key: {settings.groq_api_key[:20] if settings.groq_api_key else 'NOT SET'}...")
    print(f"huggingface_api_token: {settings.huggingface_api_token[:20] if settings.huggingface_api_token else 'NOT SET'}...")
    
    # Test get_api_keys function
    print("\nAPI Keys from get_api_keys():")
    keys = get_api_keys()
    print(f"GROQ: {keys['groq'][:20] if keys['groq'] else 'NOT SET'}...")
    print(f"HuggingFace: {keys['huggingface'][:20] if keys['huggingface'] else 'NOT SET'}...")
    
    # Test validation
    print("\nEnvironment Validation:")
    try:
        validation_result = validate_environment()
        print(validation_result)
        print("✅ Configuration is valid")
    except Exception as e:
        print(f"❌ Configuration error: {e}")

if __name__ == "__main__":
    test_config()
