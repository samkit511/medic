#!/usr/bin/env python3
"""
Test script to simulate what the frontend is doing
"""

import requests
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_frontend_request():
    print("🔍 Simulating Frontend Login Request...")
    print()
    
    # Simulate the exact request the frontend makes
    url = "https://localhost:8443/auth/login"
    headers = {
        'Content-Type': 'application/json',
        'Origin': 'http://localhost:5173'  # Frontend origin
    }
    
    # Simulate sending a fake Google ID token (this should fail with 401, not timeout)
    data = {
        "id_token": "fake.google.id.token"
    }
    
    try:
        print(f"Making POST request to: {url}")
        print(f"Headers: {headers}")
        print(f"Data: {data}")
        print()
        
        response = requests.post(
            url, 
            json=data, 
            headers=headers,
            verify=False,  # Accept self-signed cert
            timeout=30     # Give it more time
        )
        
        print(f"✅ Request completed!")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 401:
            print("\n✅ This is expected - auth endpoint is working but rejecting invalid token")
        elif response.status_code == 404:
            print("\n❌ Auth endpoint not found - this is the problem!")
        elif response.status_code == 500:
            print("\n❌ Server error - check backend logs")
        else:
            print(f"\n⚠️ Unexpected status: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - backend is too slow or not responding")
        print("💡 The backend may be loading AI models which takes time")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("💡 Server might not be running or certificate issues")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_frontend_request()
