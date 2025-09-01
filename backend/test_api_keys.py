#!/usr/bin/env python3
"""Test script to validate API keys by making actual API calls"""

import os
import requests
from config import get_api_keys

def test_groq_api():
    """Test GROQ API key"""
    keys = get_api_keys()
    groq_key = keys['groq']
    
    if not groq_key:
        print("❌ GROQ API key not found")
        return False
        
    # Test GROQ API call
    headers = {
        'Authorization': f'Bearer {groq_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'messages': [{'role': 'user', 'content': 'Hello, this is a test message.'}],
        'model': 'llama-3.1-8b-instant',
        'max_tokens': 10
    }
    
    try:
        response = requests.post('https://api.groq.com/openai/v1/chat/completions', 
                               headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            print("✅ GROQ API key is valid")
            return True
        elif response.status_code == 401:
            print(f"❌ GROQ API key is invalid: {response.json()}")
            return False
        else:
            print(f"❌ GROQ API error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ GROQ API connection error: {e}")
        return False

def test_huggingface_api():
    """Test HuggingFace API key"""
    keys = get_api_keys()
    hf_key = keys['huggingface']
    
    if not hf_key:
        print("❌ HuggingFace API key not found")
        return False
        
    # Test HuggingFace API call
    headers = {
        'Authorization': f'Bearer {hf_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'inputs': 'Hello, this is a test message.',
        'parameters': {'max_length': 10}
    }
    
    try:
        # Using a simple text generation model for testing
        response = requests.post('https://api-inference.huggingface.co/models/gpt2', 
                               headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            print("✅ HuggingFace API key is valid")
            return True
        elif response.status_code == 401:
            print(f"❌ HuggingFace API key is invalid: {response.json()}")
            return False
        else:
            print(f"❌ HuggingFace API error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ HuggingFace API connection error: {e}")
        return False

def main():
    print("=" * 50)
    print("API Keys Direct Test")
    print("=" * 50)
    
    groq_valid = test_groq_api()
    print()
    hf_valid = test_huggingface_api()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"GROQ API: {'✅ Valid' if groq_valid else '❌ Invalid'}")
    print(f"HuggingFace API: {'✅ Valid' if hf_valid else '❌ Invalid'}")
    
    if groq_valid or hf_valid:
        print("\n✅ At least one API key is working!")
    else:
        print("\n❌ Both API keys have issues!")

if __name__ == "__main__":
    main()
